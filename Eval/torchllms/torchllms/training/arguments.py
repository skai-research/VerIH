import argparse
from pathlib import Path

from torchllms.models.networks import AttentionImpl


def get_parser():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )

    # Model and tokenizer
    parser.add_argument(
        "--ckpt_paths",
        nargs="*",
        type=Path,
        default=["/data/norman_mu/models/torchllms/llama-2-7b-chat/checkpoint.00.pth"],
        help="List of model weight paths to be loaded sequentially (e.g. after partial fine-tuning). "
        "Model's params.json should reside in the same directory as the first checkpoint. "
        "Weights are allowed to not exist.",
    )
    parser.add_argument(
        "--template_config",
        type=str,
        default=None,
        help="Template config file name in torchllms/messages/configs.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--use_role_embeddings",
        action="store_true",
        help="Use role embeddings (i.e. instruction segment embeddings).",
    )
    parser.add_argument(
        "--role_embeddings_init",
        type=str,
        default="zeros",
        help="Initialization for role embeddings.",
    )

    # LoRA
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA.",
    )
    parser.add_argument(
        "--lora_embedding",
        action="store_true",
        help="Use LoRA for input embeddings.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=2,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--lora_unfrozen_params",
        nargs="*",
        type=str,
        default=[],
        help="List of LoRA parameters to unfreeze.",
    )

    # Data
    parser.add_argument(
        "--train_data_paths",
        nargs="*",
        type=str,
        default=["local:data/systemchat_converted"],
        help="Training dataset paths or HF hub names. Prefix 'local:' for local HF datasets and 'jsonl:' for local jsonl files. No prefix will pass entry to datasets.load_dataset().",
    )
    parser.add_argument(
        "--val_data_paths",
        nargs="*",
        type=str,
        default=[],
        help="Validation dataset paths (same format as train_data_paths)",
    )
    parser.add_argument(
        "--train_data_splits",
        nargs="*",
        type=str,
        default=[],
        help="Training dataset splits.",
    )
    parser.add_argument(
        "--train_num_samples",
        nargs="*",
        type=float,
        default=[],
        help="Number of samples to use from each dataset. >1 for exact count, <1 for ratio, and -1 for all samples. Defaults to all samples from all datasets.",
    )
    parser.add_argument(
        "--val_num_samples",
        nargs="*",
        type=float,
        default=[],
        help="Number of validation samples to use from each dataset. >1 for exact count, <1 for ratio, and -1 for all samples.",
    )
    parser.add_argument(
        "--val_data_splits",
        nargs="*",
        type=str,
        default=[],
        help="Validation dataset splits.",
    )
    parser.add_argument(
        "--pack_samples",
        action="store_true",
        help="Pack samples into sequences of max_seq_len.",
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        default=100,
        help="Frequency (in steps) to run validation",
    )

    # System
    parser.add_argument(
        "--attention_impl",
        type=AttentionImpl,
        default=AttentionImpl.FLASH,
        help="Attention implementation choice.",
    )
    parser.add_argument(
        "--num_data_workers",
        type=int,
        default=1,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--selective_ac_ratio",
        type=float,
        default=0,
        help="Ratio of blocks to apply selective activation checkpointing to.",
    )

    # Optimization
    parser.add_argument(
        "--micro_batch_size_per_gpu",
        type=int,
        default=4,
        help="Number of samples to process simultaneously per GPU",
    )
    parser.add_argument(
        "--loss_reduction",
        type=str,
        default="sequences",
        choices=["tokens", "sequences"],
        help="Whether to give equal weight to each token or each sequence when computing loss. (SFT only)",
    )
    parser.add_argument(
        "--fp32_logits",
        action="store_true",
        help="Upcast logits to fp32.",
    )
    parser.add_argument(
        "--gradient_accum_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--betas",
        nargs=2,
        type=float,
        default=(0.9, 0.999),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Max L2 grad norm. O for no clipping.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Total number of warmup steps",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=-1,
        help="Total number of training steps to perform. Repeats dataset as necessary. Takes priority over train_epochs and train_min_steps. -1 to use train_epochs.",
    )

    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )

    # Outputs and logging
    parser.add_argument(
        "--output_dir", type=Path, default="output", help="Where to store the model."
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=500,
        help="Frequency (full training steps) to save the model.",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=1,
        help="Maximum number of checkpoints to save.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--no_visualize_dataset", action="store_true", help="Don't show sample data"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Doesn't let me leave it blank.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="default",
        help="Wandb project name. Required if using WandB",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
    )

    return parser
