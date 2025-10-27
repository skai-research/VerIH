from typing import Optional

import torch


class DecodingCache:
    """
    A simple implementation of a cache for past values while decoding:
    - key/value vectors (kv cache)
    - role IDs
    - attention masks

    A DecodingCache is created for each (batched) generation and passed to the model in
    every forward pass. Each layer reads/writes to the same object independently.

    KV vectors are read/written *after* positional embeddings are applied.
    """

    def __init__(
        self, n_layers, max_bsz, max_seqlen, n_kv_heads, head_dim, device, dtype
    ):
        cache_shape = (n_layers, max_bsz, max_seqlen, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)

        self.role_id_cache = torch.zeros(
            (max_bsz, max_seqlen), device=device, dtype=torch.long
        )
        self.attn_mask_cache = torch.zeros(
            (max_bsz, max_seqlen), device=device, dtype=torch.long
        )
        self.is_attn_mask_cached = False
        self.next_start_pos = torch.zeros(max_bsz, device=device, dtype=torch.long)

        self.seen_tokens = [0] * n_layers  # seen_tokens always counts masked tokens
        self.max_seqlen = max_seqlen

    @torch.no_grad()
    def update_role_ids(self, role_ids: Optional[torch.Tensor]):
        """Should be called before KV cache update."""

        if role_ids is None:
            return None

        start_pos = self.seen_tokens[0]
        end_pos = start_pos + role_ids.shape[1]
        self.role_id_cache[:, start_pos:end_pos] = role_ids
        return self.role_id_cache[:, :end_pos]

    @torch.no_grad()
    def update_attn_mask(self, attn_mask: Optional[torch.Tensor]):
        """Should be called before KV cache update."""

        if attn_mask is None:
            # we want to support only passing in attn_mask for the first prefill step
            if not self.is_attn_mask_cached:
                return None

            # automatically extend for next decoding step
            self.attn_mask_cache[:, self.seen_tokens[0]] = 1
            return self.attn_mask_cache[:, : self.seen_tokens[0] + 1]

        self.is_attn_mask_cached = True
        start_pos = self.seen_tokens[0]
        end_pos = start_pos + attn_mask.shape[1]
        self.attn_mask_cache[:, start_pos:end_pos] = attn_mask
        return self.attn_mask_cache[:, :end_pos]

    @torch.no_grad()
    def update_kv(self, layer_id, k_val, v_val):
        # k_val, v_val: [B, S, H, D]

        start_pos = self.seen_tokens[layer_id]
        tgt_len = k_val.shape[1]

        self.k_cache[layer_id, :, start_pos : start_pos + tgt_len] = k_val
        self.v_cache[layer_id, :, start_pos : start_pos + tgt_len] = v_val

        k_full = self.k_cache[layer_id, :, : start_pos + tgt_len]
        v_full = self.v_cache[layer_id, :, : start_pos + tgt_len]

        self.seen_tokens[layer_id] += tgt_len

        return k_full, v_full

    def evict(self, evict_mask):
        self.k_cache = self.k_cache[:, ~evict_mask]
        self.v_cache = self.v_cache[:, ~evict_mask]
        self.role_id_cache = self.role_id_cache[~evict_mask]
        self.attn_mask_cache = self.attn_mask_cache[~evict_mask]
        self.next_start_pos = self.next_start_pos[~evict_mask]

    def is_full(self):
        return any([s >= self.max_seqlen for s in self.seen_tokens])
