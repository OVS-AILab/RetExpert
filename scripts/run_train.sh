#!/bin/bash

# set GPU devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# pretrained model path
PRETRAINED_PATH="../pretrained_model/RETFound_cfp_weights.pth"
DATA_PATH="../data/MuReD"

# start distributed training
# adapter_mode: AKU, AKU_tt
# tuning_mode: adapter (adapter, projection, head, norm), head, batchnorm, full, last_blocks
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 train.py \
    --dataset MuReD \
    --data_path ${DATA_PATH} \
    --nb_classes 20 \
    --model vit_large_patch16 \
    --finetune ${PRETRAINED_PATH} \
    --input_size 224 \
    --batch_size 16 \
    --epochs 200 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --criterion RAL \
    --adapter_mode AKU \
    --tuning_mode adapter \
    --beta_UAML 0.1 \
    --gamma_FDCM 0.1 \
    --alpha_SOA 0.1 \
    --output_dir ./output_dir/mured_retexpert_run1 \
    --seed 42