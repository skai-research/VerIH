#!/bin/bash

MODEL=${1}
VARIANT=${2:-sys_ifeval}

cd "$(dirname "$0")"
echo "Evaluating" $VARIANT
python utils/evaluation_main.py \
    --input_data inputs/${VARIANT}.jsonl \
    --input_response_data outputs/${MODEL}/${VARIANT}.jsonl \
    2>&1 | awk '!/\[nltk_data\]/'
echo ""
cd -