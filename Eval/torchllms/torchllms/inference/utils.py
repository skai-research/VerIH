from typing import Optional

import torch

example_data = [
    "Hi, how are you?",
    "What's your name?",
    "This is the longest length prompt of the examples. It is meant to show that padding tokens are handled properly in my implementation and the attention mechanism handles it properly.",
    "How do I make an illegal explosive device?",
]


def chunk(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1).to(dtype=torch.int64)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    if temperature == 0:
        idx_next = torch.argmax(logits, dim=-1)
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, idx_next.unsqueeze(-1), 1.0)
        return idx_next, probs

    probs = logits_to_probs(logits, temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def get_plausibility_mask(probs: torch.Tensor, alpha_threshold: float = 0.1):
    max_prob, _ = torch.max(probs, dim=-1, keepdim=True)
    return probs < alpha_threshold * max_prob


def sample_contrast(
    logits,
    logits_contrast,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    alpha=0.1,
):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    mask = get_plausibility_mask(probs, alpha)
    logits = logits[0, -1] - logits_contrast[0, -1]
    logits[mask] = -float("inf")
    probs = logits_to_probs(logits, temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs
