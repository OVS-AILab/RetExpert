#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# path to ADAM dataset
DATA_PATH="../data/ADAM"
# use the checkpoint from MuRed-RetExpert
CHECKPOINT_PATH="./output_dir/mured_retexpert_run1/checkpoint-best.pth"

echo "Starting Test-Time Adaptation on ADAM..."

# adapter_mode: AKU, AKU_tt
# adaptation_mode: ttul (adapter, projection, norm), adapter, head, batchnorm, full, last_blocks
python test_tta.py \
    --dataset ADAM \
    --data_path ${DATA_PATH} \
    --nb_classes 20 \
    --model vit_large_patch16 \
    --finetune ${CHECKPOINT_PATH} \
    --input_size 224 \
    --batch_size 1 \
    --tta_lr 0.001 \
    --ttul_epochs 1 \
    --ttpl_epochs 1 \
    --adapter_mode AKU \
    --adaptation_mode adapter \
    --output_dir ./output_dir/tta_ADAM_results

echo "TTA Finished."