#!/bin/bash

set -x

MODEL_DIR=${1}
VARIANT=${2}
NUM_GPU=${3}
EVAL_DIR=${4:-$VARIANT}

cd "$(dirname "$0")"
for SUITE in extraction hijacking helpful; do
    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/${OUTPUT_FOLDER}/${SUITE}.jsonl \
        --provider vllm \
        --provider_kwargs "{\"model_path\": \"${MODEL_DIR}\",  \"tensor_parallel_size\": ${NUM_GPU}, \"total_tokens\": 10240, \"response_tokens\": 8192 }" \
        --generate_kwargs "{\"temperature\": 0.0}"
done
cd -
