#!/bin/bash

# 変数の定義
CUDA_DEVICE=2
IMAGE_PATH="/mnt/npalfs01disk/users/itohlee/dataset/CACD/ID/CACD_2_outlier/55_Mark_Hamill_0017.jpg"
AGE_INIT=29
GENDER="male"
SAVE_AGED_DIR="experiment/IDP/55_Mark_Hamill_0017_gender_"
SPECIALIZED_PATH="pretrained_model/CACD_2_mark"
# SPECIALIZED_PATH="/mnt/npalfs01disk/users/itohlee/cloned/FADING/pretrained_model/finetune_double_prompt_150_random"
TARGET_AGES=(10 20 40 60 80)

# コマンドの実行
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python age_editing_IDP.py \
  --image_path $IMAGE_PATH \
  --age_init $AGE_INIT \
  --gender $GENDER \
  --save_aged_dir $SAVE_AGED_DIR \
  --specialized_path $SPECIALIZED_PATH \
  --target_ages "${TARGET_AGES[@]}"
