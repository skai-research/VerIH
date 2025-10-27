"""
Convert between HF and our custom torchllms checkpoint formats.

HF -> torchllms: reads standard HF model directory and saves `consolidated.00.pth`, copying tokenizer files and model configs.

```
python -m torchllms.models.checkpoint_converter \
    --ckpt_paths /path/to/model \
    --output /path/to/model/
```

torchllms -> HF: reads list of PyTorch weights and saves `pytorch_model.bin`, copying tokenizer files and model configs.

```
python -m torchllms.models.checkpoint_converter \
    --ckpt_paths /path/to/torchllms_model/model.pth \
    --output_dir /path/to/hf_model/ \
    --to_hf
```

torchllms -> HF also supports sequentially loading multiple checkpoints, useful for merging LoRA weights.

```
python -m torchllms.models.checkpoint_converter \
    --ckpt_paths /path/to/model/consolidated.00.pth /path/to/model/lora.pth \
    --output_dir /path/to/model/ \
    --to_hf
```

"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import glob
import json
import os
import pprint
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from torchllms import models

TOKENIZER_FILENAMES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
]


def dtype_byte_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def maybe_safe_load(file):
    if "safetensors" in file:
        with safe_open(file, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
    else:
        state_dict = torch.load(
            str(file), map_location="cpu", mmap=True, weights_only=True
        )
    return state_dict


def _from_hf(
    checkpoint, n_heads, n_kv_heads, dim, precision, head_dim=None, olmo2_arch=False
):
    if head_dim is None:
        head_dim = dim // n_heads

    dtype = getattr(torch, precision)

    def permute(w, n_head):
        if len(w.shape) == 2:
            return (
                w.view(n_head, 2, head_dim // 2, dim)
                .transpose(1, 2)
                .reshape(head_dim * n_head, dim)
            )
        else:
            return (
                w.view(n_head, 2, head_dim // 2)
                .transpose(1, 2)
                .reshape(head_dim * n_head)
            )

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    if olmo2_arch:
        weight_map |= {
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
        }

    final_result = {}
    for key, value in checkpoint.items():
        value = value.to(dtype)
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in final_result.keys():
        # Permute QK projs to account for different RoPE dimension orderings
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            final_result[key] = permute(q, n_heads)
            final_result[key.replace("wq", "wk")] = permute(k, n_kv_heads)

        if "q_norm" in key:
            q_norm = final_result[key]
            k_norm = final_result[key.replace("q_norm", "k_norm")]
            final_result[key] = permute(q_norm, n_heads)
            final_result[key.replace("q_norm", "k_norm")] = permute(k_norm, n_kv_heads)

    return final_result


def _to_hf(checkpoint, n_heads, n_kv_heads, dim, head_dim=None, olmo2_arch=False):
    if head_dim is None:
        head_dim = dim // n_heads

    def inverse_permute(w, n_head):
        if len(w.shape) == 2:
            return (
                w.view(n_head, head_dim // 2, 2, dim)
                .transpose(1, 2)
                .reshape(head_dim * n_head, dim)
            )
        else:
            return (
                w.view(n_head, head_dim // 2, 2)
                .transpose(1, 2)
                .reshape(head_dim * n_head)
            )

    reverse_weight_map = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
        "layers.{}.attention.wq.bias": "model.layers.{}.self_attn.q_proj.bias",
        "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
        "layers.{}.attention.wk.bias": "model.layers.{}.self_attn.k_proj.bias",
        "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
        "layers.{}.attention.wv.bias": "model.layers.{}.self_attn.v_proj.bias",
        "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
        "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
        "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
        "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
        "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
        "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }

    if olmo2_arch:
        reverse_weight_map |= {
            "layers.{}.attention_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
            "layers.{}.ffn_norm.weight": "model.layers.{}.post_feedforward_layernorm.weight",
            "layers.{}.attention.q_norm.weight": "model.layers.{}.self_attn.q_norm.weight",
            "layers.{}.attention.k_norm.weight": "model.layers.{}.self_attn.k_norm.weight",
        }

    hf_checkpoint = {}
    for key, value in checkpoint.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
            layer_num = re.search(r"\d+", key).group(0)
            hf_key = reverse_weight_map[abstract_key].format(layer_num)
        else:
            hf_key = reverse_weight_map[key]

        hf_checkpoint[hf_key] = value

    for key in hf_checkpoint.keys():
        # Un-permute QK projs to account for different RoPE dimension orderings
        if "q_proj" in key:
            q = hf_checkpoint[key]
            k = hf_checkpoint[key.replace("q_proj", "k_proj")]
            hf_checkpoint[key] = inverse_permute(q, n_heads)
            hf_checkpoint[key.replace("q_proj", "k_proj")] = inverse_permute(
                k, n_kv_heads
            )

        if "q_norm" in key:
            q_norm = hf_checkpoint[key]
            k_norm = hf_checkpoint[key.replace("q_norm", "k_norm")]
            hf_checkpoint[key] = inverse_permute(q_norm, n_heads)
            hf_checkpoint[key.replace("q_norm", "k_norm")] = inverse_permute(
                k_norm, n_kv_heads
            )

    return hf_checkpoint


@torch.inference_mode()
def convert_from_hf_checkpoint(checkpoint: Path, output_dir: Path, precision: str):
    if checkpoint.is_dir():
        checkpoint_dir = checkpoint
        # Load the json file containing weight mapping
        if (checkpoint_dir / "pytorch_model.bin.index.json").is_file():
            with open(checkpoint_dir / "pytorch_model.bin.index.json") as json_map:
                bin_index = json.load(json_map)
            bin_files = {
                checkpoint_dir / bin for bin in bin_index["weight_map"].values()
            }

        # Might be using safetensors instead
        elif (checkpoint_dir / "model.safetensors.index.json").is_file():
            with open(checkpoint_dir / "model.safetensors.index.json") as json_map:
                bin_index = json.load(json_map)
            bin_files = {
                checkpoint_dir / bin for bin in bin_index["weight_map"].values()
            }

        elif (checkpoint_dir / "model.safetensors").is_file():
            bin_files = {checkpoint_dir / "model.safetensors"}

        elif (checkpoint_dir / "pytorch_model.bin").is_file():
            bin_files = {checkpoint_dir / "pytorch_model.bin"}

        elif bin_files := glob.glob(str(checkpoint_dir / "*.safetensors")):
            bin_files = {Path(f) for f in bin_files}

        elif bin_files := glob.glob(str(checkpoint_dir / "*.bin")):
            bin_files = {Path(f) for f in bin_files}

        elif bin_files := glob.glob(str(checkpoint_dir / "*.pt")):
            bin_files = {Path(f) for f in bin_files}

        else:
            raise ValueError(f"Could not find any model weights in {checkpoint_dir}")
    else:
        checkpoint_dir = checkpoint.parent
        bin_files = {checkpoint}

    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    print("Loaded model config:")
    pprint.pprint(config)

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = maybe_safe_load(str(file))
        merged_result.update(state_dict)

    final_result = _from_hf(
        merged_result,
        n_heads=config["num_attention_heads"],
        n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
        dim=config["hidden_size"],
        precision=precision,
        head_dim=config.get("head_dim", None),
        olmo2_arch="olmo2" in config["architectures"][0].lower(),
    )

    output_dir = output_dir or checkpoint_dir
    output_path = output_dir / "consolidated.00.pth"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving checkpoint to {output_path}")
    torch.save(final_result, output_path)

    for file in TOKENIZER_FILENAMES + ["config.json"]:
        src = checkpoint_dir / file
        dst = output_dir / file
        if src.is_file():
            if dst.is_file() and src.samefile(dst):
                continue
            print(f"Copying {src} -> {dst}")
            shutil.copy(src, dst)
        else:
            print(f"Missing {file}")

    with open(output_dir / "params.json", "w") as f:
        _config = {
            "dim": config["hidden_size"],
            "multiple_of": 256,
            "n_heads": config["num_attention_heads"],
            "n_kv_heads": config.get(
                "num_key_value_heads", config["num_attention_heads"]
            ),
            "ffn_dim_multiplier": config["intermediate_size"] / config["hidden_size"],
            "n_layers": config["num_hidden_layers"],
            "vocab_size": config["vocab_size"],
            "attn_proj_bias": "qwen" in config["architectures"][0].lower(),
            "tie_word_embeddings": config.get("tie_word_embeddings", False),
        }
        if head_dim := config.get("head_dim", None):
            _config["head_dim"] = head_dim
        if rope_scaling := config.get("rope_scaling", None):
            _config["rope_scaling"] = rope_scaling
        if norm_eps := config.get("rms_norm_eps", None):
            _config["norm_eps"] = norm_eps
        if rope_theta := config.get("rope_theta", None):
            _config["rope_theta"] = rope_theta
        if "olmo2" in config["architectures"][0].lower():
            _config["olmo2_arch"] = True

        print("\nSaving model config:")
        pprint.pprint(_config)
        json.dump(_config, f)


@torch.inference_mode()
def convert_to_hf_checkpoint(ckpt_paths: list[Path], output_dir: Path):
    print("Loading model and tokenizer...")
    main_ckpt_dir = ckpt_paths[-1].parent

    with open(main_ckpt_dir / "params.json", "r") as f:
        params = json.load(f)

    print("Loaded model config:")
    pprint.pprint(params)

    lora_args = None
    if (main_ckpt_dir / "lora_args.json").exists():
        with open(main_ckpt_dir / "lora_args.json", "r") as f:
            lora_args = json.load(f)

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

    merge_device = (
        "cuda" if torch.cuda.is_available() and lora_args is not None else "cpu"
    )
    model = models.load_model_weights(
        ckpt_paths,
        model=model,
        precision="bfloat16",
        device=merge_device,
    )

    with torch.device(merge_device):
        models.utils.init_meta_params(model)

    if lora_args is not None:
        # move to GPU for faster weight merging
        model.to(merge_device)
        # merge LoRA weights into the original model weights for efficient inference
        model.apply(models.lora.replace_lora_with_linear)
        if lora_args.get("lora_embedding", False):
            model.apply(models.lora.replace_lora_with_embedding)
        model.to("cpu")

    hf_checkpoint = _to_hf(
        model.state_dict(),
        params["n_heads"],
        params.get("n_kv_heads", params["n_heads"]),
        params["dim"],
        params.get("head_dim", None),
        olmo2_arch=params.get("olmo2_arch", False),
    )

    output_dir = output_dir or main_ckpt_dir
    output_path = output_dir / "pytorch_model.bin"

    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / "pytorch_model.bin.index.json", "w") as f:
        total_size = sum(
            v.numel() * dtype_byte_size(v.dtype) for v in hf_checkpoint.values()
        )
        json.dump(
            {
                "weight_map": {k: "pytorch_model.bin" for k in hf_checkpoint.keys()},
                "metadata": {"total_size": total_size},
            },
            f,
        )

    print(f"\nSaving checkpoint to {output_path}")
    torch.save(hf_checkpoint, output_path)

    for file in TOKENIZER_FILENAMES + ["config.json"]:
        src = main_ckpt_dir / file
        dst = output_dir / file
        if src.is_file():
            if dst.is_file() and src.samefile(dst):
                continue
            print(f"Copying {src} -> {dst}")
            shutil.copy(src, dst)
        else:
            print(f"Missing {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between checkpoint formats.")
    parser.add_argument(
        "--ckpt_paths",
        nargs="*",
        type=Path,
        default=[
            Path("/data/norman_mu/models/torchllms/llama-2-7b-chat/checkpoint.00.pth")
        ],
        help="Model weight paths to be loaded sequentially (e.g. the result of partial fine-tuning). Model config should reside in the same directory as the first checkpoint. Weights may not exist.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output dir for the converted checkpoint. Defaults to input directory if unspecified.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        help="Precision to use for the converted checkpoint.",
    )
    parser.add_argument(
        "--to_hf",
        action="store_true",
    )
    args = parser.parse_args()

    if args.to_hf:
        convert_to_hf_checkpoint(args.ckpt_paths, args.output_dir)
    else:
        assert len(args.ckpt_paths) == 1, "hf checkpoint should just be one folder."
        convert_from_hf_checkpoint(args.ckpt_paths[0], args.output_dir, args.precision)
