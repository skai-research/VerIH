#!/bin/bash

set -x

MODEL_DIR=${1}
export VARIANT=${2}
NUM_GPU=${3}

# set max_token in src/generation_utils.py create_and_inference_with_vllm()
# set response_token in src/generation_utils.py inference_with_vllm()
cd "$(dirname "$0")"

python utils/evaluation/eval.py generators \
  --use_vllm \
  --model_name_or_path ${MODEL_DIR} \
  --model_input_template_path_or_name hf \
  --tasks ${VARIANT} \
  --report_output_path ./outputs/${OUTPUT_FOLDER}/${VARIANT}.jsonl \
  --save_individual_results_path ./outputs/${OUTPUT_FOLDER}/${VARIANT}_eval.json
cd -
