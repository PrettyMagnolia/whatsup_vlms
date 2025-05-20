class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import os
import torch
import pickle
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
import pycocotools.mask as mask_util

def load_edges(edge_path, image_shape):
    # return np.ones(image_shape[:2], dtype=np.uint8)
    # return np.zeros(image_shape[:2], dtype=np.uint8)·
    # return np.full(image_shape[:2], 255, dtype=np.uint8)
    if os.path.exists(edge_path):
        with open(edge_path, 'rb') as f:
            combined_edges = pickle.load(f)
        rle = {'size': combined_edges['size'], 'counts': combined_edges['counts']}
        mask = mask_util.decode(rle)
        return mask
    else:
        print(f"Edges file not found: {edge_path}")
        return np.ones(image_shape[:2], dtype=np.uint8)

def get_objects_sense(image, edge_path, preprocess):
    edges = load_edges(edge_path, image.shape[1:])
    edges = torch.as_tensor(edges).unsqueeze(0).float() * 255
    edges = preprocess(edges)
    return edges


def fill_edges_to_objects(edges_mask):
    """
    将只包含边缘的掩码（1 表示物体边缘）转换为物体区域。
    1) 先将 edges_mask > 0 转为 bool 类型
    2) 用 binary_fill_holes 填充内部区域
    3) 用 label 对各连通区域打 label ID
    返回：labels，其值为 0 表示背景，1,2,... 表示不同物体。
    """
    # edges_mask 可能是 (H, W) 的 0/1 矩阵
    bool_mask = edges_mask > 0
    # 填充物体内部
    filled_mask = binary_fill_holes(bool_mask)
    # 连通组件标记
    labels = label(filled_mask)
    return labels

def patch_to_object_mapping(labels, patch_size):
    """
    给定按 label 分好的物体区域，以及指定的 patch 大小，
    统计每个 patch 内哪个 label(对象)面积占比最大，
    返回一个 2D 数组 patch_map，形状是 (n_patch_row, n_patch_col)，
    其中每个元素是该 patch 最占主导的对象 ID。
    """
    H, W = labels.shape
    n_patch_row = (H + patch_size - 1) // patch_size
    n_patch_col = (W + patch_size - 1) // patch_size
    
    patch_map = np.zeros((n_patch_row, n_patch_col), dtype=int)
    
    for pr in range(n_patch_row):
        for pc in range(n_patch_col):
            # 每个 patch 的图像坐标范围
            y_start = pr * patch_size
            y_end   = min((pr+1)*patch_size, H)
            x_start = pc * patch_size
            x_end   = min((pc+1)*patch_size, W)
            
            patch_region = labels[y_start:y_end, x_start:x_end]
            if patch_region.size == 0:
                continue
            
            # 计算出现最多的 label
            # (不计算 label=0，因为那是背景)
            vals, counts = np.unique(patch_region, return_counts=True)
            # 去掉背景
            bg_idx = np.where(vals == 0)
            vals = np.delete(vals, bg_idx)
            counts = np.delete(counts, bg_idx)
            
            if len(vals) > 0:                
                dominant_label = vals[np.argmax(counts)]
            else:
                dominant_label = 0  # 如果patch里没有任何对象
                
            patch_map[pr, pc] = dominant_label
    
    return patch_map

def generate_object_attention_mask(patch_map):
    patch_map_flat = patch_map.flatten()  # (n_patches,)
    # add CLS
    patch_map_flat = np.concatenate((np.array([0]), patch_map_flat, ))  # (n_patches + 1,)
    n_patches = patch_map_flat.shape[0]
    
    object_id_row = patch_map_flat[:, None]
    object_id_col = patch_map_flat[None, :]
    
    same_object = (object_id_row == object_id_col)
    
    return same_object.astype(np.float32)  # 1.0表示可以看见，0.0表示不能

def get_visible_matrix(edges_mask_batch):
    num_heads = 12 
    if edges_mask_batch is None:
        return None
    
    # 获取原张量所在的设备
    original_device = edges_mask_batch.device

    batch_size = edges_mask_batch.size(0)
    attn_masks = []

    for b in range(batch_size):
        edges_mask = edges_mask_batch[b, 0]  # (B, 1, H, W) -> (H, W)
        
        # Convert edges_mask tensor to NumPy
        edges_mask_numpy = edges_mask.cpu().detach().numpy()
        
        # 使用NumPy数组进行必要的运算
        labels = fill_edges_to_objects(edges_mask_numpy)
        patch_map = patch_to_object_mapping(labels, patch_size=32)
        attn_mask_numpy = generate_object_attention_mask(patch_map)

        # Convert back to tensor
        attn_mask_tensor = torch.tensor(attn_mask_numpy)

        # Move back to original device
        attn_mask_tensor = attn_mask_tensor.to(original_device)

        # 构造attention mask
        attn_mask_new = torch.where(
            attn_mask_tensor.bool(),
            torch.zeros_like(attn_mask_tensor, dtype=torch.float32),    # 相同object，mask 0（正常）
            torch.full_like(attn_mask_tensor, fill_value=-1e9, dtype=torch.float32)  # 不同object，mask为-∞
        )

        
        attn_masks.extend([attn_mask_new] * num_heads)  # 扩展到num_heads个

    # Return stacked tensor
    return torch.stack(attn_masks, dim=0)

def get_visible_matrix_v2(image, edge_path, preprocess, patch_size=32):
    # 读取 edges
    edges_mask = get_objects_sense(image, edge_path, preprocess).squeeze(0)  # (1, H, W) -> (H, W)
    # fill + 连通组件分析
    labels = fill_edges_to_objects(edges_mask)
    # 按照 patch_size，对 labels 做统计，得到 patch -> object
    patch_map = patch_to_object_mapping(labels, patch_size)

    visible_matrix = generate_object_attention_mask(patch_map)

    visible_matrix = torch.tensor(visible_matrix, dtype=torch.float32)

    # 构造attention mask
    visible_matrix = torch.where(
        visible_matrix.bool(),
        torch.zeros_like(visible_matrix, dtype=torch.float32),    # 相同object，mask 0（正常）
        torch.full_like(visible_matrix, fill_value=-1e9, dtype=torch.float32)  # 不同object，mask为-∞
    )
    return visible_matrix
