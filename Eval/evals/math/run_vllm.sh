#!/bin/bash

set -x

MODEL_DIR=${1}
VARIANT=${2}
NUM_GPU=${3}

# export THINK_TIMES=1
cd "$(dirname "$0")"
python -m torchllms.inference.generate \
    --input_path inputs/${VARIANT}.jsonl \
    --output_path outputs/${OUTPUT_FOLDER}/${VARIANT}.jsonl \
    --provider vllm \
    --provider_kwargs "{\"model_path\": \"${MODEL_DIR}\",  \"tensor_parallel_size\": ${NUM_GPU}, \"total_tokens\": 35000, \"response_tokens\": 32768 }" \
    --generate_kwargs "{\"temperature\": 0.0}"
cd -
