export N_GPUS=4
export ROLLOUT_TP_SIZE=2
export TEMPLATED_INPUT=1
export BASE_MODEL="Qwen/Qwen3-8B"
export DATA_DIR="dataset/qwen3-verih-cot_syshint"
export EXPERIMENT_NAME="Qwen3-8B-GRPO-01R-2048-verih"
unset ROCR_VISIBLE_DEVICES
bash ./scripts/train_grpo.sh
