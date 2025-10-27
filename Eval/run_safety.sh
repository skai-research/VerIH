export NUM_GPU=2
export OPENAI_API_KEY=""
# ["Base", "CoT", "CoTGuardRules", "CoTSysHintGuardRules"]
export TIMESTAMP="$(date +%m%d%H%M%S)"
echo "MODEL_PATH=$MODEL_PATH, MODE=$TEST_MODE, $TIMESTAMP"

RES_DIR=$(realpath "./${TIMESTAMP}.res")
echo "Result file $RES_DIR"
export OUTPUT_FOLDER="${MODEL_NAME}-${TEST_MODE}-${TIMESTAMP}"
echo "MODEL_PATH=$MODEL_PATH, MODE=$TEST_MODE, $TIMESTAMP" >> "$RES_DIR"
cd ./evals

# Safety
bash safety-eval/run_vllm.sh "$MODEL_PATH" harmbench "$NUM_GPU"
bash read.sh "$OUTPUT_FOLDER" harmbench "safety-eval" >> "$RES_DIR"

bash safety-eval/run_vllm.sh "$MODEL_PATH" wildjailbreak:benign "$NUM_GPU"
bash read.sh "$OUTPUT_FOLDER" wildjailbreak:benign "safety-eval" >> "$RES_DIR"
bash safety-eval/run_vllm.sh "$MODEL_PATH" wildjailbreak:harmful "$NUM_GPU"
bash read.sh "$OUTPUT_FOLDER" wildjailbreak:harmful "safety-eval" >> "$RES_DIR"

bash tensortrust/run_vllm.sh "$MODEL_PATH" tensortrust "$NUM_GPU"
bash tensortrust/read.sh "$OUTPUT_FOLDER" tensortrust >> "$RES_DIR"