import json
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import yaml
from loguru import logger
from transformers import AutoTokenizer

from torchllms import models
from torchllms.distributed import get_rank
from torchllms.messages import configs, tokenization


def setup_model_and_tokenizer(
    ckpt_paths: List[str],
    template_config: Optional[str] = None,
    device: str = "cuda",
    precision: str = "bfloat16",
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    """Set up model and tokenizer for inference on a single GPU.

    Args:
        ckpt_paths (List[str]): Checkpoint path(s) to load sequentially
        template_config (Optional[str], optional): Chat template config to use instead of tokenizer's chat template. Defaults to None.
        device (str, optional): Defaults to "cuda".
        precision (str, optional): Defaults to "bfloat16".
    """
    ckpt_paths = [Path(p) for p in ckpt_paths]
    main_ckpt_dir = ckpt_paths[-1].parent
    tokenizer = AutoTokenizer.from_pretrained(main_ckpt_dir, trust_remote_code=True)

    if template_config is not None:
        with resources.files(configs).joinpath(template_config).open() as file:
            config = yaml.safe_load(file)
            template_config = tokenization.TemplateConfig(**config)

    with open(main_ckpt_dir / "params.json", "r") as f:
        params = json.load(f)

    lora_args = None
    if (main_ckpt_dir / "lora_args.json").exists():
        with open(main_ckpt_dir / "lora_args.json", "r") as f:
            lora_args = json.load(f)

    if device == "cpu" or precision == "float32":
        params["attention_impl"] = "sdpa"

    if model_kwargs is not None:
        params.update(model_kwargs)

    model_params = models.networks.ModelParams(**params)

    with torch.device("meta"):
        model = models.networks.Transformer.from_params(model_params)

    if lora_args is not None:
        replace_linear_with_lora = models.lora.make_replace_linear_with_lora(
            rank=lora_args["lora_rank"],
            alpha=lora_args["lora_alpha"],
            dropout=lora_args["lora_dropout"],
        )
        model.apply(replace_linear_with_lora)

        if lora_args.get("lora_embedding", False):
            replace_embedding_with_lora = models.lora.make_replace_embedding_with_lora(
                rank=lora_args["lora_rank"],
                alpha=lora_args["lora_alpha"],
                dropout=lora_args["lora_dropout"],
            )
            model.apply(replace_embedding_with_lora)

    models.load_model_weights(
        ckpt_paths,
        model=model,
        precision=precision,
        device=device,
    )

    with torch.device(device):
        models.utils.init_meta_params(model)

    if lora_args is not None:
        # merge LoRA weights into the original model weights for efficient inference
        model.apply(models.lora.replace_lora_with_linear)
        if lora_args.get("lora_embedding", False):
            model.apply(models.lora.replace_lora_with_embedding)

    return model, tokenizer, template_config


def load_model_weights(
    ckpt_paths: List[Path],
    model: nn.Module,
    precision: Optional[Literal["bfloat16", "float16", "float32"]] = None,
    device: Optional[torch.device] = None,
):
    for ckpt_path in ckpt_paths:
        if ckpt_path.exists():
            if get_rank() == 0:
                logger.opt(colors=True).info(
                    f"<green>Loading checkpoint from:</green> {ckpt_path}"
                )
            checkpoint = torch.load(
                str(ckpt_path), mmap=True, weights_only=False, map_location=device
            )
            result = model.load_state_dict(checkpoint, strict=False, assign=True)
            if len(result.missing_keys) > 0:
                missing_keys_str = "\n".join(
                    collate_param_names(list(result.missing_keys))
                )
                if get_rank() == 0:
                    logger.opt(colors=True).warning(
                        f"<yellow>Params missing for sub-modules:</yellow>\n<dim>{missing_keys_str}</dim>"
                    )

            elif len(result.unexpected_keys) > 0:
                unexpected_keys_str = "\n".join(
                    collate_param_names(list(result.unexpected_keys))
                )
                if get_rank() == 0:
                    logger.opt(colors=True).warning(
                        f"<yellow>Unexpected params for sub-modules:</yellow>\n<dim>{unexpected_keys_str}</dim>"
                    )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if precision is not None:
        precision = getattr(torch, precision)
    else:
        precision = next(iter(checkpoint.values())).dtype

    model = model.to(dtype=precision)
    return model


def init_meta_params(model):
    meta_params_names = []
    for name, module in model.named_modules():
        meta_params = [
            n
            for n, p in module.named_parameters(recurse=False)
            if p.device.type == "meta"
        ]
        meta_buffers = [
            n for n, p in module.named_buffers(recurse=False) if p.device.type == "meta"
        ]
        if len(meta_params) > 0 or len(meta_buffers) > 0:
            for param in meta_params:
                meta_params_names.append(f"{name}.{param}")
            for buffer in meta_buffers:
                meta_params_names.append(f"{name}.{buffer}")
            module.to_empty(device="cpu", recurse=False)
            init_module(module)

    meta_params_names = "\n".join(collate_param_names(meta_params_names))
    if get_rank() == 0:
        logger.opt(colors=True).info(
            f"<green>Re-initialized all modules containing meta params/buffers:</green>\n<dim>{meta_params_names}</dim>"
        )


def merge_checkpoints(ckpt_paths: List[Path]):
    state = {}
    for ckpt_path in ckpt_paths:
        if not ckpt_path.exists():
            continue
        checkpoint = torch.load(str(ckpt_path), mmap=True, weights_only=True)
        state.update(checkpoint)
    return state


def collate_param_names(param_names: List[str]):
    outputs = []
    patterns = defaultdict(list)

    for param in param_names:
        parts = param.split(".")
        if len(parts) > 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_idx = parts[1]
            rest = ".".join(parts[2:])
            patterns[rest].append(layer_idx)
        else:
            outputs.append(param)

    for pattern in patterns.keys():
        layer_idxs = ",".join(patterns[pattern])
        outputs.append(f"layers.{{{layer_idxs}}}.{pattern}")

    return outputs


def init_module(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
        return

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight.data)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    else:
        raise ValueError(f"Unknown initialization for module type: {type(module)}")


# Attention masking utils copied in from HuggingFace Transformers.
# 2 new dims going from 2d to 4d: placeholder heads dim and query length

# For prefill with a attn mask [0, 1, 1] to_4d_and_causal would return:
# [[[[-inf, -inf, -inf],
#    [-inf, 0,   -inf],
#    [-inf, 0,   0]]]]

# Then for decoding, to_4d_and_causal would return:
# [[[[-inf, 0, 0, 0]]]]


def to_4d_and_causal(
    attention_mask_2d: torch.Tensor,
    query_length: int,
    dtype: torch.dtype,
    key_value_length: int,
) -> torch.Tensor:
    """
    Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
    key_value_length) shape and by adding a large negative bias to not-attended positions, and adds a causal mask.
    """
    input_shape = (attention_mask_2d.shape[0], query_length)

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    past_key_values_length = key_value_length - query_length
    causal_4d_mask = _make_causal_mask(
        input_shape,
        dtype,
        device=attention_mask_2d.device,
        past_key_values_length=past_key_values_length,
    )

    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_attn_mask = _expand_mask(
        attention_mask_2d, dtype, tgt_len=input_shape[-1]
    ).to(attention_mask_2d.device)

    expanded_attn_mask = causal_4d_mask.masked_fill(
        expanded_attn_mask.bool(), torch.finfo(dtype).min
    )

    return expanded_attn_mask


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        zeros = torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device)
        mask = torch.cat([zeros, mask], dim=-1)

    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )
