export NUM_GPU=2

# ["Base", "CoT", "CoTSysHint"]
export TIMESTAMP="$(date +%m%d%H%M%S)"
echo "MODEL_PATH=$MODEL_PATH, MODE=$TEST_MODE, $TIMESTAMP"

RES_DIR=$(realpath "./${TIMESTAMP}.res")
echo "Result file $RES_DIR"
export OUTPUT_FOLDER="${MODEL_NAME}-${TEST_MODE}-${TIMESTAMP}"
echo "MODEL_PATH=$MODEL_PATH, MODE=$TEST_MODE, $TIMESTAMP" >> "$RES_DIR"
cd ./evals

# IF & IH
bash run_vllm.sh "$MODEL_PATH" ifeval "$NUM_GPU"
bash ifeval/read.sh "$OUTPUT_FOLDER" ifeval >> "$RES_DIR"

bash run_vllm.sh "$MODEL_PATH" ifbench "$NUM_GPU"
bash ifbench/read.sh "$OUTPUT_FOLDER" ifbench >> "$RES_DIR"

# Phi-4-mini does not have tool calling ability
bash iheval/run_vllm.sh "$MODEL_PATH" "$NUM_GPU"
bash read.sh "$OUTPUT_FOLDER" "iheval" >> "$RES_DIR"

# General - only for CoT mode
bash run_vllm.sh "$MODEL_PATH" mmlu "$NUM_GPU"
bash read.sh "$OUTPUT_FOLDER" mmlu >> "$RES_DIR"

bash math/run_vllm.sh "$MODEL_PATH" math500 "$NUM_GPU" math
bash read.sh "$OUTPUT_FOLDER" math500 math >> "$RES_DIR"
