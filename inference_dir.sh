#!/bin/bash

# 変数の定義
CUDA_DEVICE=2
DATA_PATH="/mnt/npalfs01disk/users/itohlee/dataset/CACD/ID/CACD_2_"
SAVE_AGED_DIR="experiment/IDP/CACD_2_mark"
SPECIALIZED_PATH="pretrained_model/CACD_2_mark"
# SPECIALIZED_PATH="pretrained_model/finetune_double_prompt_150_random"
TARGET_AGES=(10 20 30 45 60 80)
TEST_BATCH_SIZE=4
TEST_WORKERS=4

# コマンドの実行
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python age_editing_IDP_dir.py \
  --data_path $DATA_PATH \
  --save_aged_dir $SAVE_AGED_DIR \
  --specialized_path $SPECIALIZED_PATH \
  --target_ages "${TARGET_AGES[@]}" \
  --test_batch_size $TEST_BATCH_SIZE \
  --test_workers $TEST_WORKERS
