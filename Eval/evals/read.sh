#!/bin/bash

MODEL=${1}
VARIANT=${2}
EVAL_DIR=${3:-$VARIANT}

cd "$(dirname "$0")"
echo "Evaluating" $VARIANT
python ${EVAL_DIR}/evaluate.py \
    --response ${EVAL_DIR}/outputs/${MODEL}/${VARIANT}.jsonl \
    --variant $VARIANT
echo ""
cd -

