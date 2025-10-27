import os

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device, non_blocking=True)
        except Exception:
            output[k] = v
    return output


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def wrap_fsdp(model):
    fsdp_kwargs = {
        "reshard_after_forward": True,
        # "offload_policy": CPUOffloadPolicy(),
    }

    if not model.params.tie_word_embeddings:
        fully_shard(model.output, **fsdp_kwargs)

    for m in reversed(list(model.layers)):
        fully_shard(m, **fsdp_kwargs)

    if not model.params.tie_word_embeddings:
        fully_shard(model.tok_embeddings, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)
