# nohup python open_clip_test.py \
#     --model-name ViT-B-32 \
#     --pretrained /home/yifei/code/open_clip/logs/Objects-CLIP-CC3M/checkpoints/epoch_best.pt \
#     --output-dir /mnt/user_data/yifei/outputs/whatsup/ \
#     --exp-name Objects-CLIP-CC3M-obj \
#     --use_obj_token \
#     --use_img_token_vm \
#     --img_token_vm_layers 6 \
#     > objects-clip-test.log 2>&1 &

nohup python open_clip_test.py \
    --model-name ViT-B-32 \
    --pretrained /home/yifei/code/open_clip/logs/Objects-CLIP-ALL-new-objw01/checkpoints/epoch_best.pt \
    --output-dir /mnt/user_data/yifei/outputs/whatsup/ \
    --exp-name Objects-CLIP-ALL-new-objw01-raw \
    > objects-clip-test.log 2>&1 &

# kill -9 $(pgrep -f open_clip_test.py)
