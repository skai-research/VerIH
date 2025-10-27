# Training Scripts

Example training scripts from different projects can be found in the [scripts](../../scripts) folder. Most experiments were conducted with 4x A100-80GB GPUs, though DPO/SimPO may require 8x GPUs if not using activation checkpointing.

## Model Weight Format

1. Convert HF weights before training:
```bash
huggingface-cli download meta-llama/Llama-3.1-8b-instruct --local-dir checkpoints/llama3.1_8b_instruct

python -m torchllms.models.checkpoint_converter --ckpt_paths checkpoints/llama3.1_8b_instruct --output checkpoints/llama3.1_8b_instruct
```

The converted weights (`consolidated.00.pth`), model config (`params.json`) and tokenizer files will be at `checkpoints/llama3.1_8b_instruct/`.

## Custom Tokenization

We use a custom YAML-based template system instead of HF's chat templates. The templates are defined in `torchllms/messages/configs/`.

For example, with LLaMA models we use `llama3_instruct.yaml` which defines:
- Special tokens (BOS, EOS, etc.)
- Role tokens for system/user/assistant messages
- Token masking rules for training

## Training Data Formats

### SFT Data
Chat conversation data in JSONL format:
```json
{
  "messages": [
    {"role": "system", "content": "Optional system message"},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

### Preference Data (DPO/SimPO)
Paired conversations with preferred/rejected responses:
```json
{
  "chosen": [
    {"role": "system", "content": "Optional system message"}, 
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Preferred response"}
  ],
  "rejected": [
    {"role": "system", "content": "Optional system message"},
    {"role": "user", "content": "User message"}, 
    {"role": "assistant", "content": "Less preferred response"}
  ]
}
```

## Training Modes

See [torchllms/training/arguments.py](../arguments.py) for all supported training arguments.

### Supervised Fine-Tuning (SFT)
Standard cross-entropy training on conversation data:

```bash
torchrun --nproc_per_node 1 torchllms/training/trainers/sft.py \
    --ckpt_paths checkpoints/llama3.1_8b_instruct/consolidated.00.pth \
    --template_config llama3_instruct.yaml \
    --lr 1e-6 \
    --lr_scheduler cosine \
    --warmup_steps 5 \
    --wd 0 \
    --train_data_paths jsonl:examples.jsonl \
    --train_epochs 1 \
    --max_seq_len 1024 \
    --micro_batch_size_per_gpu 2 \
    --gradient_accum_steps 2 \
    --output_dir outputs/sft_example
```

### Direct Preference Optimization (DPO)
Trains on preference data using DPO loss:

```bash
torchrun --nproc_per_node 1 torchllms/training/trainers/dpo.py \
    --ckpt_paths checkpoints/llama3.1_8b_instruct/consolidated.00.pth \
    --template_config llama3_instruct.yaml \
    --lr 5e-7 \
    --lr_scheduler cosine \
    --dpo_beta 0.01 \
    --train_data_paths jsonl:examples.jsonl \
    --train_epochs 1 \
    --max_seq_len 1024 \
    --micro_batch_size_per_gpu 1 \
    --gradient_accum_steps 8 \
    --output_dir outputs/dpo_example
```

### Simple Preference Optimization (SimPO)
Alternative preference learning using margin-based loss:

```bash
torchrun --nproc_per_node 1 torchllms/training/trainers/simpo.py \
    --ckpt_paths checkpoints/llama3.1_8b_instruct/consolidated.00.pth \
    --template_config llama3_instruct.yaml \
    --lr 5e-7 \
    --lr_scheduler cosine \
    --simpo_beta 1.0 \
    --simpo_gamma 1.0 \
    --train_data_paths jsonl:examples.jsonl \
    --train_epochs 1 \
    --max_seq_len 1024 \
    --micro_batch_size_per_gpu 1 \
    --gradient_accum_steps 8 \
    --output_dir outputs/simpo_example
```

## LoRA Training

All training modes support LoRA by adding:
```bash
--lora \
--lora_rank 16 \
--lora_alpha 16 \
--lora_dropout 0.0
```

## Common Arguments

All trainers share these core arguments:

- `--ckpt_paths`: Path(s) to model checkpoint(s)
- `--template_config`: YAML config for chat template (e.g. `llama3_instruct.yaml`)
- `--max_seq_len`: Maximum sequence length
- `--micro_batch_size_per_gpu`: Batch size per GPU
- `--gradient_accum_steps`: Number of gradient accumulation steps
- `--output_dir`: Directory to save checkpoints

### Data Arguments

`--train_data_paths` supports:
- HuggingFace datasets: `HuggingFaceH4/ultrachat_200k`
- Local jsonl files: `jsonl:/path/to/data.jsonl`
- Local text files: `text:/path/to/data.txt`
- Local HF datasets: `local:/path/to/dataset`

### Optimization Arguments

- `--loss_reduction`: How to compute mean loss
  - `tokens`: `sum(per_token_losses) / num_tokens`
  - `sequences`: `sum(per_sequence_losses) / num_sequences`

Effective batch size = `micro_batch_size_per_gpu * gradient_accum_steps * num_gpus`

## Distributed Training

All scripts support distributed training via torchrun. Adjust:

- `--nproc_per_node`: Number of GPUs to use
- `--micro_batch_size_per_gpu`: Per-GPU batch size
- `--gradient_accum_steps`: Steps before optimizer update

The effective batch size will be: `micro_batch_size_per_gpu * gradient_accum_steps * nproc_per_node`