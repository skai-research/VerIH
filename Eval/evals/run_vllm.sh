#!/bin/bash

set -x

MODEL_DIR=${1}
VARIANT=${2}
NUM_GPU=${3}
EVAL_DIR=${4:-$VARIANT}

cd "$(dirname "$0")"
python -m torchllms.inference.generate \
    --input_path ${EVAL_DIR}/inputs/${VARIANT}.jsonl \
    --output_path ${EVAL_DIR}/outputs/${OUTPUT_FOLDER}/${VARIANT}.jsonl \
    --provider vllm \
    --provider_kwargs "{\"model_path\": \"${MODEL_DIR}\",  \"tensor_parallel_size\": ${NUM_GPU}, \"total_tokens\": 10240, \"response_tokens\": 8192 }" \
    --generate_kwargs "{\"temperature\": 0.0}"
cd -
