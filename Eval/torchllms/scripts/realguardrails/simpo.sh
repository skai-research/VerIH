#!/bin/bash

set -x

EPOCHS=${1:-1}
LR=${2:-2e-5}
SEED=${3:-1}
BASE=${4:-llama3_8b}
CONFIG=${5:-llama3_instruct.yaml}
BETA=${6:-1}
GAMMA=${7:-1}

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node 8 torchllms/training/trainers/simpo.py \
    --ckpt_paths /data/norman_mu/code/torchllms/outputs/${BASE}/model_final.pth \
    --template_config ${CONFIG} \
    --lr ${LR} \
    --lr_scheduler cosine \
    --clip_grad_norm 1.0 \
    --warmup_steps 50 \
    --wd 0 \
    --betas 0.9 0.999 \
    --eps 1e-8 \
    --simpo_beta ${BETA} \
    --simpo_gamma ${GAMMA} \
    --train_data_paths jsonl:data/preferencemix.jsonl \
    --train_epochs ${EPOCHS} \
    --max_seq_len 4096 \
    --fp32_logits \
    --micro_batch_size_per_gpu 1 \
    --gradient_accum_steps 16 \
    --selective_ac_ratio 0.5 \
    --save_freq 999999 \
    --print_freq 1 \
    --no_visualize_dataset \
    --seed ${SEED} \
    --wandb \
    --wandb_group final \
    --wandb_project system \
    --wandb_entity normster \
    --output_dir outputs/system/final-simpo-${BASE}_lr=${LR}_epochs=${EPOCHS}_beta=${BETA}_gamma=${GAMMA}_seed=${SEED}