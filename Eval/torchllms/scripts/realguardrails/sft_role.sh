#!/bin/bash

set -x

DATA=${1}
EPOCHS=${2:-1}
SEED=${3:-1}
BASE=${4:-llama3_8b}
CONFIG=${5:-llama3_instruct.yaml}
INIT=${6:-"zeros"}

IFS=',' read -ra items <<< "$1"

args=()
for item in ${items[@]}; do
    args+=("jsonl:data/$item.jsonl")
done

BATCH_SIZE=$((4 * ACCUM_STEPS))

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node 4 torchllms/training/trainers/sft.py \
    --ckpt_paths /data/norman_mu/models/torchllms/${BASE}/consolidated.00.pth \
    --use_role_embeddings \
    --role_embeddings_init $INIT \
    --template_config ${CONFIG} \
    --lr 2e-5 \
    --lr_scheduler cosine \
    --warmup_steps 200 \
    --wd 0 \
    --betas 0.9 0.999 \
    --eps 1e-8 \
    --train_data_paths "${args[@]}" \
    --train_epochs ${EPOCHS} \
    --max_seq_len 4096 \
    --micro_batch_size_per_gpu 1 \
    --gradient_accum_steps 32 \
    --loss_reduction sequences \
    --clip_grad_norm 1.0 \
    --save_freq 999999 \
    --print_freq 1 \
    --no_visualize_dataset \
    --seed ${SEED} \
    --wandb \
    --wandb_group final \
    --wandb_project system \
    --wandb_entity normster \
    --output_dir outputs/system/final-${BASE}_role=${INIT}_data=${DATA}_epochs=${EPOCHS}_seed=${SEED}

