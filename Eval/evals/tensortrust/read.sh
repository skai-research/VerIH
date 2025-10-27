#!/bin/bash

MODEL=${1}
VARIANT=${2}
EVAL_DIR=${3:-$VARIANT}

cd "$(dirname "$0")"
echo "Evaluating" $VARIANT
python evaluate.py --outputs_dir outputs/${MODEL}
echo ""
cd -

