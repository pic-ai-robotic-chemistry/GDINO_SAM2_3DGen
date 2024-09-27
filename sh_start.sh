python ./Grounded-SAM-2/gen_bbox_mask_for_image.py \
--text_prompt "Xiaolin-Robot." \
--img_path "./test_imgs/Xiaolin-Robot.jpg" \
--sam2_checkpoint "./Grounded-SAM-2/checkpoints/sam2_hiera_large.pt" \
--sam2_model_config "sam2_hiera_l.yaml" \
--grounding_dino_checkpoint "./Grounded-SAM-2/gdino_checkpoints/checkpoint.pth" \
--grounding_dino_config "./Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
--output_dir './outputs/Xiaolin-Robot'

python ./Fourier123/infer_lgm.py big \
--resume ./Fourier123/pretrained/model_fp16_fixrot.safetensors \
--workspace ./outputs/Xiaolin-Robot \
--test_path ./outputs/Xiaolin-Robot/mask_rgba.png

#CUDA_VISIBLE_DEVICES=0 python ./Fourier123/main.py \
#--config ./Fourier123/configs/image.yamconda
#input=./outputs/mask_rgba.png \
#save_path=backpack \
#load=outputs/backpack/backpack_rgba.ply
#
#CUDA_VISIBLE_DEVICES=0 python /Fourier123/see.py --config ./Fourier123configs/image.yaml workspace=outputs/backpack load=logs/backpack_model.ply
