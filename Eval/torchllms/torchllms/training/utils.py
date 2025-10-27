import gc
import glob
import json
import os
import shutil
from functools import partial
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from torchllms.distributed import get_rank
from torchllms.models.utils import collate_param_names, merge_checkpoints
from torchllms.models.networks import AttentionImpl


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, AttentionImpl):
            return obj.value

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def save_args_and_params(args, params, main_ckpt_dir):
    for file in glob.glob(str(main_ckpt_dir / "*.json")):
        shutil.copy(file, args.output_dir)

    with open(args.output_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, cls=JSONEncoder)

    with open(args.output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, cls=JSONEncoder)

    if args.lora:
        with open(args.output_dir / "lora_args.json", "w") as f:
            lora_args = {
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_embedding": args.lora_embedding,
            }
            json.dump(lora_args, f, indent=2)


def get_full_params(model):
    sharded_sd = model.state_dict()
    cpu_state = {}

    for param_name, sharded_param in sharded_sd.items():
        if sharded_param.is_cpu:
            sharded_param = sharded_param.to("cuda")
        full_param = sharded_param.full_tensor()
        if get_rank() == 0:
            cpu_state[param_name] = full_param.cpu()
        else:
            del full_param

    return cpu_state


def save_model(model, save_dir, step_name="final", max_checkpoints=1, frozen_params=[]):
    if save_dir is None:
        return

    cpu_state = get_full_params(model)

    if get_rank() == 0:
        cpu_state = {k: v for k, v in cpu_state.items() if k not in frozen_params}
        save_params_str = "\n".join(collate_param_names(cpu_state.keys()))
        logger.opt(colors=True).info(
            f"<green>Saving unfrozen params:</green>\n<dim>{save_params_str}</dim>"
        )

        ckpt_name = f"model_{step_name}.pth"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(cpu_state, os.path.join(save_dir, ckpt_name))

        all_ckpts = sorted(glob.glob(str(save_dir / "model_*.pth")))
        for f in all_ckpts[:-max_checkpoints]:
            logger.opt(colors=True).warning(
                f"<red>Removing old checkpoint:</red> <white>{f}</white>"
            )
            os.remove(f)

    dist.barrier()


def apply_fsdp_checkpointing(model, block, p):
    """
    Apply selective activation checkpointing.

    Selectivity is defined as a percentage p, which means we apply ac
    on p of the total blocks. p is a floating number in the range of
    [0, 1].

    Some examples:
    p = 0: no ac for all blocks. same as `fsdp_activation_checkpointing=False`
    p = 1: apply ac on every block. i.e. "full ac".
    p = 1/2: [ac, no-ac, ac, no-ac, ...]
    p = 1/3: [no-ac, ac, no-ac,   no-ac, ac, no-ac,   ...]
    p = 2/3: [ac, no-ac, ac,    ac, no-ac, ac,    ...]
    Since blocks are homogeneous, we make ac blocks evenly spaced among
    all blocks.

    Implementation:
    For a given ac ratio p, we should essentially apply ac on every "1/p"
    blocks. The first ac block can be as early as the 0th block, or as
    late as the "1/p"th block, and we pick the middle one: (0.5p)th block.
    Therefore, we are essentially to apply ac on:
    (0.5/p)th block, (1.5/p)th block, (2.5/p)th block, etc., and of course,
    with these values rounding to integers.
    Since ac is applied recursively, we can simply use the following math
    in the code to apply ac on corresponding blocks.
    """
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, block):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=selective_checkpointing,
    )


def check_weights(ckpt_paths: List[Path], model: nn.Module):
    changed = []
    unchanged = []
    new = []

    cpu_state = get_full_params(model)

    if get_rank() == 0:
        state = merge_checkpoints(ckpt_paths)
        for name, param in cpu_state.items():
            if name in state:
                if state[name].shape != param.shape:
                    logger.opt(colors=True).warning(
                        f"<red>Shape mismatch for {name}: {state[name].shape} != {param.shape}</red>"
                    )
                if torch.allclose(state[name], param):
                    unchanged.append(name)
                else:
                    changed.append(name)
            else:
                new.append(name)

        if len(new) > 0:
            new = "\n".join(collate_param_names(new))
            logger.opt(colors=True).info(
                f"<yellow>New params:</yellow>\n<dim>{new}</dim>"
            )

        if len(changed) > 0:
            changed = "\n".join(collate_param_names(changed))
            logger.opt(colors=True).info(
                f"<yellow>Changed params:</yellow>\n<dim>{changed}</dim>"
            )

        if len(unchanged) > 0:
            unchanged = "\n".join(collate_param_names(unchanged))
            logger.opt(colors=True).info(
                f"<yellow>Unchanged params:</yellow>\n<dim>{unchanged}</dim>"
            )

    dist.barrier()


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
