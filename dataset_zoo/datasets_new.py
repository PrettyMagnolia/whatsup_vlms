import pdb
import os
import json
import subprocess
import copy
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from torchvision.datasets.utils import download_url

from .perturbations import TextShuffler
from .retrieval import pre_caption
from .utils import get_obj_token_mask, get_img_token_vm_mask


class BaseDataset(Dataset):
    def __init__(self, root_dir, annotation_file="annotations", image_preprocess=None, **kwargs):
        self.root_dir = root_dir
        self.image_preprocess = image_preprocess

        self.annotation_file = os.path.join(root_dir, f"{annotation_file}.jsonl")
        self.image_dir = os.path.join(root_dir, "images")
        
        with open(self.annotation_file, "r") as f:
            self.dataset = [json.loads(line) for line in f]

        self.use_obj_token = kwargs.get("use_obj_token", False)
        self.use_img_token_vm = kwargs.get("use_img_token_vm", False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        if 'sub_dir' in sample:
            image_path = os.path.join(self.image_dir, sample['sub_dir'], f"{sample['image_id']}.jpg")
        else:
            image_path = os.path.join(self.image_dir, f"{sample['image_id']}.jpg")
        image = Image.open(image_path).convert('RGB')

        ori_img_size = resize_img_size = image.size[::-1]
        if self.image_preprocess:
            image = self.image_preprocess(image)
            resize_img_size = image.shape[-2:]

        attn_mask = self.get_attn_mask(
            bboxes=sample["bboxes"][:10],
            ori_img_size=ori_img_size,
            resize_img_size=resize_img_size
        )

        item = edict({"image_options": [image], "attn_mask": [attn_mask], "caption_options": sample["caption_options"]})
        return item

    def get_attn_mask(self, bboxes, ori_img_size, resize_img_size):
        attn_mask = None

        if self.use_obj_token:
            attn_mask = get_obj_token_mask(
                bboxes=bboxes,
                image_original_size=ori_img_size, 
                image_resize_size=resize_img_size, 
                patch_size=32
            )

        if self.use_img_token_vm:
            vm_mask = get_img_token_vm_mask(
                bboxes=bboxes,
                image_original_size=ori_img_size, 
                image_resize_size=resize_img_size, 
                patch_size=32
            )

            if self.use_obj_token:
                # 合并到 attn_mask 中
                # 去掉 [CLS] 对应的第一行和第一列
                vm_mask = vm_mask[1:, 1:]

                # 替换 obj_token_mask 最右下角的 img_token 部分
                img_token_size = vm_mask.shape[0]
                attn_mask[-img_token_size:, -img_token_size:] = vm_mask
            else:
                attn_mask = vm_mask

        # 如果没有使用任何 token mask，默认返回一个 -1
        return attn_mask if attn_mask is not None else torch.tensor([-1])
        

class VG_Relation(BaseDataset):
    def __init__(self, root_dir, annotation_file="annotations", image_preprocess=None, **kwargs):
        super().__init__(root_dir, annotation_file, image_preprocess, **kwargs)
        self.all_relations = [sample['relation_name'] for sample in self.dataset]

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)

        all_relations = np.array(self.all_relations)

        result_records = []
        # Log the accuracy of all relations
        for relation in np.unique(all_relations):
            relation_mask = (all_relations == relation)
            if relation_mask.sum() == 0:
                continue
            result_records.append({
                "Relation": relation,
                "Accuracy": correct_mask[relation_mask].mean(),
                "Count": relation_mask.sum(),
                "Dataset": "Visual Genome Relation"
            })
        result_records.append({
            "Relation": "Overall",
            "Accuracy": np.mean(correct_mask),
            "Count": len(correct_mask),
            "Dataset": "Visual Genome Relation"
        })
        return result_records, np.mean(correct_mask)
    

class VG_Attribution(BaseDataset):
    def __init__(self, root_dir, annotation_file="annotations", image_preprocess=None, **kwargs):
        super().__init__(root_dir, annotation_file, image_preprocess, **kwargs)
        self.all_attributes = [f"{sample['attributes'][0]}_{sample['attributes'][1]}" for sample in self.dataset]

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        result_records = []
        all_attributes = np.array(self.all_attributes)
        for attr in np.unique(all_attributes):
            attr_mask = (all_attributes == attr)
            if attr_mask.sum() < 25:
                continue
            result_records.append({
                "Attributes": attr,
                "Accuracy": correct_mask[attr_mask].mean(),
                "Count": attr_mask.sum(),
                "Dataset": "Visual Genome Attribution"
            })
        result_records.append({
            "Attributes": "Overall",
            "Accuracy": np.mean(correct_mask),
            "Count": len(correct_mask),
            "Dataset": "Visual Genome Attribution"
        })
        return result_records, np.mean(correct_mask)
    

class COCO_Order(BaseDataset):
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        
        else:
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        records = [{"Accuracy": np.mean(correct_mask)}]
        return records, np.mean(correct_mask)
    

class Flickr_Order(BaseDataset):
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        
        else:
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        records = [{"Accuracy": np.mean(correct_mask)}]
        return records, np.mean(correct_mask)


class COCO_Spatial(BaseDataset):
    def __init__(self, root_dir, annotation_file="annotations", image_preprocess=None, **kwargs):
        super().__init__(root_dir, annotation_file, image_preprocess, **kwargs)
        self.all_prepositions = [sample['relation_name'] for sample in self.dataset]

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        # print(metrics['Accuracy']*100)

        all_prepositions = np.array(self.all_prepositions)

        prepositions = list(set(self.all_prepositions))
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above', 'top': 'bottom', 'bottom': 'top'}

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        #print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "COCO-Spatial"
            })
        result_records.append({
            "Preposition": "Overall",
            "Accuracy": np.mean(correct_mask),
            "Count": len(correct_mask),
            "Dataset": "COCO-Spatial"
        })
        return result_records, np.mean(correct_mask)


class GQA_Spatial(BaseDataset):
    def __init__(self, root_dir, annotation_file="annotations", image_preprocess=None, **kwargs):
        super().__init__(root_dir, annotation_file, image_preprocess, **kwargs)
        self.all_prepositions = [sample['relation_name'] for sample in self.dataset]

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        # print(metrics['Accuracy']*100)

        all_prepositions = np.array(self.all_prepositions)
        
        prepositions = list(set(self.all_prepositions)) + ['below', 'bottom', 'front']
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {'left': 'right', 'right': 'left', 'behind': 'front', 'front': 'behind', 'above': 'below', 'below': 'above', 'bottom': 'top', 'top': 'bottom'}

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        #print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "GQA-Spatial"
            })
        result_records.append({
            "Preposition": "Overall",
            "Accuracy": np.mean(correct_mask),
            "Count": len(correct_mask),
            "Dataset": "GQA-Spatial"
        })
        return result_records, np.mean(correct_mask)


class Whatsup(BaseDataset):
    def __init__(self, root_dir, annotation_file="annotations", image_preprocess=None, **kwargs):
        super().__init__(root_dir, annotation_file, image_preprocess, **kwargs)

        self.all_prepositions = [sample['relation_name'] for sample in self.dataset]
        self.image_dir = os.path.join(root_dir, "images", annotation_file)
        self.eval_dict = self.pred_dict = {
            (sample['obj1_name'], sample['obj2_name']): {
                'left': False,
                'right': False,
                'on': False,
                'under': False,
                'in-front': False,
                'behind': False
            }
            for sample in self.dataset
        }

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 4, i.e. first caption is right, next three captions are wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        print("Individual accuracy: {}".format(metrics['Accuracy']*100))

        prepositions = ['on', 'under', 'front', 'behind', 'left', 'right']
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        for i, (sample, correct) in enumerate(zip(self.dataset, correct_mask)):
            prep = list(set(prepositions).intersection(set(sample['caption_options'][preds[i]].split())))
            gold_prep = list(set(prepositions).intersection(set(sample['caption_options'][0].split())))

            prep = prep[0]
            gold_prep = gold_prep[0]
            prep_counts[gold_prep][prep] += 1

            self.pred_dict[(sample['obj1_name'], sample['obj2_name'])][sample['relation_name']] = prep
            self.eval_dict[(sample['obj1_name'], sample['obj2_name'])][sample['relation_name']] = correct
        
        pair_correct = 0
        set_correct = 0
        for obj_pair, correct_dict in self.eval_dict.items():
            if correct_dict['left'] and correct_dict['right']:
                pair_correct += 1
            if correct_dict['on'] and correct_dict['under']:
                pair_correct += 1
            if correct_dict['in-front'] and correct_dict['behind']:
                pair_correct += 1
            if sum(correct_dict.values()) == 4:
                set_correct += 1
        pair_accuracy = pair_correct*100/(len(self.dataset)/2)
        set_accuracy = set_correct*100/(len(self.dataset)/4)
        print("Pair accuracy: {}".format(pair_accuracy))
        print("Set accuracy: {}".format(set_accuracy))
        all_prepositions = np.array(self.all_prepositions)

        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "Whatsup"
            })
        result_records.append({
            "Preposition": "Overall",
            "Accuracy": np.mean(correct_mask),
            "Count": len(correct_mask),
            "Dataset": "Whatsup"
        })
        return result_records, np.mean(correct_mask)


class Sugarcrepe(BaseDataset):
    def __init__(self, root_dir, annotation_file="annotations", image_preprocess=None, **kwargs):
        super().__init__(root_dir, annotation_file, image_preprocess, **kwargs)

        self.all_types = [sample['type'] for sample in self.dataset]

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)

        all_types = np.array(self.all_types)

        result_records = []
        for attr in np.unique(all_types):
            attr_mask = (all_types == attr)
            if attr_mask.sum() == 0:
                continue
            result_records.append({
                "Attributes": attr,
                "Accuracy": correct_mask[attr_mask].mean(),
                "Count": attr_mask.sum(),
                "Dataset": "Sugarcrepe"
            })
        result_records.append({
            "Attributes": "Overall",
            "Accuracy": np.mean(correct_mask),
            "Count": len(correct_mask),
            "Dataset": "Sugarcrepe"
        })
        return result_records, np.mean(correct_mask)
    

class VL_CheckList(BaseDataset):
    def __init__(self, root_dir, annotation_file="annotations", image_preprocess=None, **kwargs):
        super().__init__(root_dir, annotation_file, image_preprocess, **kwargs)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)

        all_types = np.array(self.all_types)

        result_records = []
        for attr in np.unique(all_types):
            attr_mask = (all_types == attr)
            if attr_mask.sum() == 0:
                continue
            result_records.append({
                "Attributes": attr,
                "Accuracy": correct_mask[attr_mask].mean(),
                "Count": attr_mask.sum(),
                "Dataset": "VL_CheckList"
            })
        result_records.append({
            "Attributes": "Overall",
            "Accuracy": np.mean(correct_mask),
            "Count": len(correct_mask),
            "Dataset": "VL_CheckList"
        })
        return result_records, np.mean(correct_mask)
    
