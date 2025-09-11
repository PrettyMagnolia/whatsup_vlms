import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset_new
from misc import seed_all, _default_collate, save_scores

from open_clip import create_model_and_transforms
from model_zoo.clip_models import CLIPWrapper

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--num_workers", default=24, type=int)
    parser.add_argument("--model-name", type=str, default='ViT-B-32')
    parser.add_argument("--pretrained", type=str, default='/mnt/shared/models/open_clip/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin')
    parser.add_argument("--seed", default=1, type=int)
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: True)", default=True)
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--exp-name", type=str, default="open_clip_test")
    parser.add_argument("--use_obj_token", default=False, action="store_true", help="use object token")
    parser.add_argument("--use_img_token_vm", default=False, action="store_true", help="use image token visible matrix")
    parser.add_argument("--img_token_vm_layers", default=3, type=int, help="use image token visible matrix layers")
    return parser.parse_args()

    
def main(args):
    seed_all(args.seed)

    model, _, image_preprocess = create_model_and_transforms(model_name=args.model_name, pretrained=args.pretrained, device=args.device)
    model = model.eval()
    model = CLIPWrapper(model, args.device, use_obj_tokens=args.use_obj_token, img_token_vm_layers=args.img_token_vm_layers)

    datasets = [
        # "VG_Attribution",
        # "VG_Relation",
        # "COCO_Order",
        # "Flickr_Order",
        # "COCO_Spatial_One",
        # "COCO_Spatial_Two",
        # "GQA_Spatial_One",
        # "GQA_Spatial_Two",
        # "Whatsup_A",
        # "Whatsup_B",
        # "Sugarcrepe",
        "VL_CheckList",
    ]
    
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for dataset_name in datasets:
        output_file = os.path.join(args.output_dir, f"{dataset_name}.csv")
        if os.path.exists(output_file):
            print(f"Results for {dataset_name} already exist. Skipping...")
            # continue

        dataset = get_dataset_new(dataset_name, image_preprocess=image_preprocess, download=args.download, use_obj_token=args.use_obj_token, use_img_token_vm=args.use_img_token_vm)
        
        # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
        collate_fn = _default_collate if image_preprocess is None else None
        
        joint_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        scores = model.get_retrieval_scores_batched(joint_loader)
        result_records, total_acc = dataset.evaluate_scores(scores)
        
        for record in result_records:
            record.update({"Model": args.model_name, "Dataset": dataset_name, "Seed": args.seed})
        
        df = pd.DataFrame(result_records)
        print(f"Saving results to {output_file}")
        if os.path.exists(output_file):
            all_df = pd.read_csv(output_file, index_col=0)
            all_df = pd.concat([all_df, df])
            all_df.to_csv(output_file)

        else:
            df.to_csv(output_file)
            
        if args.save_scores:
            save_scores(scores, args)
    

if __name__ == "__main__":
    args = config()
    main(args)
