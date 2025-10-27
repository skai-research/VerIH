# torchllms: simple LLM training/inference in PyTorch

With contributions from [Jonathan Lu](https://github.com/jonathanlu31) and [Michael Lavery](https://github.com/thavens).

## Overview

`torchllms` is a research-oriented library for fine-tuning LLMs with an emphasis on readability and hackability. Settings like LR schedule, token masking, chat template format, cross-entropy reduction, sequence packing, mixed LoRA/full fine-tuning, etc. are all easily configurable with minimal indirection and confusion.

Modeling and training code are mostly contained within single files: [torchllms/models/networks.py](torchllms/models/networks.py) and [torchllms/training/trainers/sft.py](torchllms/training/trainers/sft.py).

This library uses FSDP 2.0 for single-GPU or multi-GPU training on a single node.
We support Llama 3, Qwen 2.5, and OLMo 2 models, which means many other models may also be incidentally supported.

We also implement basic inference code, using KV caching and batched decoding, for use with custom architectures/inference strategies (e.g. instructional segment embeddings and classifier-free guidance). If you are just fine-tuning a standard model, you should convert the resulting weights back to the standard HuggingFace format with `torchllms/models/checkpoint_converter.py` and use vLLM for inference.

This repo uses a custom format for model weights than the standard HuggingFace format, based on the [original Llama implementation](https://github.com/meta-llama/llama/blob/main/llama/model.py). You can convert between the two formats with [torchllms/models/checkpoint_converter.py](torchllms/models/checkpoint_converter.py).

We also use a custom chat template system, specified in [torchllms/messages/tokenization.py](torchllms/messages/tokenization.py), which records message role IDs. There are slight differences in the exact format between our library and official Jinja2 chat templates published on HF, so to match training/inference templates the training scripts automatically copy over the corresponding `tokenizer_config.json` file from [torchllms/messages/configs](torchllms/messages/configs) based on the `--template_config` argument (e.g. `llama3_instuct.yaml` -> `llama3_instuct.json`, etc.).

## Installation

1. Install the `torchllms` library with:

```bash
pip install -e .
```

2. [Optional] Install FlashAttention2 following instructions at: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). Defaults to PyTorch's built-in attention operator if not installed.

## Usage

The general workflow for using this library looks something like:

1. Download HF weights.
2. Convert HF weights to torchllms format with [torchllms/models/checkpoint_converter.py](torchllms/models/checkpoint_converter.py)
3. Finetune the model with [torchllms/training/trainers/sft.py](torchllms/training/trainers/sft.py), or included DPO and SimPO scripts.
4. Convert back to HF format with `checkpoint_converter.py`.

### Converting from HF weights
Assuming we are using Llama-3 8B Instruct:

First, download and convert the HF weights. Weights that you have converted will have the path `model_dir/consolidated.00.pth` whereas the training checkpoints will have the path `model_dir/model_final.pth` or `model_dir/model_{checkpoint_number}.pth`.

```bash
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir outputs/llama3.2_1b_instruct --exclude "*.pth"
python -m torchllms.models.checkpoint_converter --ckpt_paths outputs/llama3.2_1b_instruct --output outputs/llama3.2_1b_instruct
```

### Training

Finetuning is supported for local/remote HuggingFace datasets, as well as local .jsonl files or text files. For a simple sanity check, try finetuning on Jonathan's [conditional lowercase dataset](https://github.com/jonathanlu31/conditional_lowercase) which trains the model to respond normally/in lowercase/in uppercase when the system message is empty/"ll"/"cps".

```bash
torchrun --nproc_per_node 1 torchllms/training/trainers/sft.py \
    --ckpt_paths outputs/llama3.2_1b_instruct/consolidated.00.pth \
    --template_config llama3_instruct.yaml \
    --lr 1e-4 \
    --lr_scheduler cosine \
    --warmup_steps 20 \
    --wd 0 \
    --betas 0.9 0.999 \
    --eps 1e-8 \
    --train_data_paths jsonl:data/ultra_lowercase_train_sample.jsonl \
    --train_num_samples 1000 \
    --train_epochs 1 \
    --max_seq_len 2048 \
    --micro_batch_size_per_gpu 2 \
    --gradient_accum_steps 2 \
    --loss_reduction sequences \
    --clip_grad_norm 1.0 \
    --print_freq 1 \
    --output_dir outputs/lowercase
```

LoRA can be enabled with the `--lora` flag. This freezes the LLM weights and only optimizes trainable low-rank adaptaion matrices for each linear layer, reducing memory usage and checkpoint size. Only the low-rank matrices are saved, along with a `lora_args.json` config file.

```bash
torchrun --nproc_per_node 1 torchllms/training/trainers/sft.py \
    --ckpt_paths outputs/llama3.2_1b_instruct/consolidated.00.pth \
    --template_config llama3_instruct.yaml \
    --lr 1e-3 \
    --lr_scheduler constant \
    --warmup_steps 20 \
    --wd 0.01 \
    --eps 1e-8 \
    --lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --train_data_paths jsonl:data/examples/ultra_lowercase_train_sample.jsonl \
    --train_num_samples 1000 \
    --train_epochs 1 \
    --max_seq_len 2048 \
    --micro_batch_size_per_gpu 2 \
    --gradient_accum_steps 2 \
    --print_freq 1 \
    --output_dir outputs/lora_lowercase
```

See [torchllms/training/trainers/TRAINERS.md](torchllms/training/trainers/TRAINERS.md) more details.

### Chat with model

After training, you can test the model by chatting with it:

```bash
python -m torchllms.inference.chat --ckpt_paths outputs/llama3.2_1b_instruct/consolidated.00.pth
```

Enter `!!reset` as a user message to reset the chat.

When chatting with a model that has been finetuned with LoRA, first load the original weights and then the LoRA weights:

```bash
python -m torchllms.inference.chat --ckpt_paths outputs/llama3.2_1b_instruct/consolidated.00.pth outputs/lora_lowercase/model_final.pth
```

### Convert back to HF

For evaluation we recommend converting to HF format for inference speed.
```bash
python -m torchllms.models.checkpoint_converter --ckpt_paths outputs/lowercase/model_final.pth --output outputs/lowercase --to_hf
```

To merge LoRA weights with the original weights and convert to Huggingface format:
```bash
python -m torchllms.models.checkpoint_converter --ckpt_paths outputs/llama3.2_1b_instruct/consolidated.00.pth outputs/lora_lowercase/model_final.pth --output outputs/lora_lowercase --to_hf
```

You can also chat with models in HF format:

```bash
python -m torchllms.inference.chat_huggingface --model_path outputs/lowercase
```