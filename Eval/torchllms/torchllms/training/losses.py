from typing import Literal, Union

import torch
import torch.nn.functional as F

from torchllms.training import data


def cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_reduction: str = Union[Literal["tokens"], Literal["sequences"]],
) -> torch.Tensor:
    """
    Compute cross entropy loss, ignoring ignored labels.
    Handles case where all labels are ignored.
    """
    batch_size, seq_len, vocab_size = logits.shape
    loss_tokens = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len)

    mask = labels != data.IGNORE_ID

    if loss_reduction == "tokens":
        loss_valid = loss_tokens[mask]
        size = loss_valid.numel()

        if size == 0:
            return torch.tensor(0.0, device=loss_tokens.device)

        return loss_valid.sum() / size

    elif loss_reduction == "sequences":
        loss_tokens[~mask] = 0.0
        label_counts = mask.sum(dim=-1)
        loss_sequences = loss_tokens.sum(dim=-1) / label_counts

        return loss_sequences.mean()


def make_loss_fn(args):
    def sft_loss(logits, labels):
        if args.fp32_logits:
            logits = logits.float()
        return cross_entropy(logits, labels, loss_reduction=args.loss_reduction)

    return sft_loss
