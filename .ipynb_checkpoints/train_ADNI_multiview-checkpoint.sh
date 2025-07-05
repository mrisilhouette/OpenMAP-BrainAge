#!/bin/bash

gpu_ids="0"
sex="all_gender"
split_json_train="./data/${sex}_train_file.csv"
split_json_eval="./data/${sex}_valid_file.csv"

lr_drop=10

data_augmentation=1


to_save_path="./trained_model_${sex}_down_resolution_with_aug_lr_drop_${lr_drop}"
file="train_${sex}_down_resolution_with_aug_lr_drop_${lr_drop}_log.txt"



CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python mainADNI_multiview.py \
    --data_dir "./data" \
    --split_json_train ${split_json_train} \
    --split_json_eval ${split_json_eval} \
    --img_size 128 128 30 \
    --epochs 200 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --to_save_path ${to_save_path} \
    --seed 3401 \
    --data_augmentation ${data_augmentation} \
    --down_resolution 1 \
    --lr_drop ${lr_drop} \
    --trunk_pretrained_path "./hpt_pretrained_model/trunk.pth"\
    --image_encoder_pretrained_path "./resnet_pretrained_model/r3d18_KM_200ep.pth"\
    --share_image_encoder 1 \
    --use_modality_tokens 0 \
    --freeze_trunk 0 \
    > ${file}
