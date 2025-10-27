# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from enum import Enum
from typing import Optional

try:
    from flash_attn import flash_attn_func
except ImportError:
    print("flash_attn not found, using native PyTorch implementation")
    flash_attn_func = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from torchllms.messages import Role
from torchllms.models import utils
from torchllms.models.cache import DecodingCache


class AttentionImpl(Enum):
    EAGER = "eager"
    FLASH = "flash"
    SDPA = "sdpa"


class ModelParams(BaseModel):
    dim: int = 4096
    head_dim: Optional[int] = None
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float = 8 / 3  # re-defined from original impl
    tie_word_embeddings: bool = False
    norm_eps: float = 1e-5
    max_seq_len: int = 4096
    attn_proj_bias: bool = False
    rope_theta: float = 10000.0
    rope_scaling: dict = {}
    use_role_embeddings: bool = False
    role_embeddings_init: str = "zeros"
    olmo2_arch: bool = False
    attention_impl: AttentionImpl = AttentionImpl.FLASH

    model_config = {"extra": "ignore"}

    def model_post_init(self, __context):
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.attention_impl == AttentionImpl.FLASH and not flash_attn_func:
            self.attention_impl = AttentionImpl.SDPA


class RoleEmbeddings(nn.Embedding):
    def __init__(self, params: ModelParams):
        self.init = params.role_embeddings_init
        super().__init__(len(Role), params.dim)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init == "zeros":
            nn.init.zeros_(self.weight)
        elif self.init.startswith("gaussian"):
            std = float(self.init.split(":")[1])
            nn.init.normal_(self.weight, std=std)
        else:
            raise ValueError(f"Unknown role embeddings init: {self.init}")


def _bias_or_norm(name):
    name = name.lower()

    if "bias" in name or "norm" in name or "ln" in name:
        return True

    return False


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        nn.init.ones_(self.weight.data)

    def forward(self, x):
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.weight


# from: https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        factor: int = 1,
        high_freq_factor: int = 1,
        low_freq_factor: int = 1,
        original_max_position_embeddings: int = 8192,
        rope_type: str = "default",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type
        self.factor = factor
        self.high_freq_factor = high_freq_factor
        self.low_freq_factor = low_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.rope_init()

    # TODO: delete this once all our recipes are moved off of FSDP1 since we
    # no longer need to explicitly name our param init method reset_parameters
    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # From: https://github.com/huggingface/transformers/blob/37ea04013b34b39c01b51aeaacd8d56f2c62a7eb/src/transformers/modeling_rope_utils.py#L310
        if self.rope_type == "llama3":
            low_freq_wavelen = self.old_context_len / self.low_freq_factor
            high_freq_wavelen = self.old_context_len / self.high_freq_factor

            wavelen = 2 * math.pi / theta
            # wavelen < high_freq_wavelen: do nothing
            # wavelen > low_freq_wavelen: divide by factor
            theta = torch.where(wavelen > low_freq_wavelen, theta / self.factor, theta)
            # otherwise: interpolate between the two, using a smooth factor
            smooth_factor = (self.old_context_len / wavelen - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * theta / self.factor + smooth_factor * theta
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            theta = torch.where(is_medium_freq, smoothed_inv_freq, theta)
        elif self.rope_type != "default":
            raise ValueError(f"Unknown RoPE type: {self.rope_type}")

        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


def _eager_attention(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    output_scores: bool = False,
):
    bsz, qlen, n_heads, head_dim = xq.shape
    _, seqlen, n_kv_heads, _ = xk.shape
    dim = n_heads * head_dim

    # manual implementation of scaled dot product attention
    xq = torch.einsum("bshd->bhsd", xq)
    xk = torch.einsum("bshd->bhsd", xk)
    xv = torch.einsum("bshd->bhsd", xv)

    if n_heads != n_kv_heads:
        kv_repeats = n_heads // n_kv_heads
        xk = xk.repeat_interleave(kv_repeats, dim=1)
        xv = xv.repeat_interleave(kv_repeats, dim=1)

    if attn_mask is None:
        attn_mask = utils._make_causal_mask(
            (bsz, qlen),
            xq.dtype,
            device=xq.device,
            past_key_values_length=seqlen - qlen,
        )
    else:
        attn_mask = utils.to_4d_and_causal(attn_mask, qlen, xq.dtype, seqlen)

    scores = torch.einsum("bhqd,bhkd->bhqk", xq, xk) / math.sqrt(head_dim)

    scores = scores + attn_mask
    weights = F.softmax(scores, dim=-1)

    output = torch.einsum("bhqk,bhkd->bqhd", weights, xv)
    output = output.reshape(bsz, qlen, dim)

    if output_scores:
        return output, scores

    return output


def _flash_attention(xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor):
    bsz, qlen, n_heads, head_dim = xq.shape
    dim = n_heads * head_dim

    output = flash_attn_func(xq, xk, xv, causal=True)
    output = output.reshape(bsz, qlen, dim)
    return output


def _sdpa_attention(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
):
    bsz, qlen, n_heads, head_dim = xq.shape
    _, seqlen, n_kv_heads, _ = xk.shape
    dim = n_heads * head_dim

    xq = torch.einsum("bshd->bhsd", xq)
    xk = torch.einsum("bshd->bhsd", xk)
    xv = torch.einsum("bshd->bhsd", xv)

    if attn_mask is not None:
        # convert 2d to 4d and apply causal mask
        attn_mask = utils.to_4d_and_causal(attn_mask, qlen, xq.dtype, seqlen)
        is_causal = False
    else:
        # don't need causal masking for decoding
        is_causal = seqlen == qlen

    output = torch.nn.functional.scaled_dot_product_attention(
        xq,
        xk,
        xv,
        attn_mask=attn_mask,
        is_causal=is_causal,
        enable_gqa=n_kv_heads != n_heads,
    )
    output = torch.einsum("bhsd->bshd", output)
    output = output.reshape(bsz, qlen, dim)
    return output


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.params = params
        self.layer_id = layer_id
        self.rope = RotaryPositionalEmbeddings(
            params.head_dim,
            params.max_seq_len,
            params.rope_theta,
            **params.rope_scaling,
        )

        self.wq = nn.Linear(
            params.dim,
            params.n_heads * params.head_dim,
            bias=params.attn_proj_bias,
        )
        self.wk = nn.Linear(
            params.dim,
            params.n_kv_heads * params.head_dim,
            bias=params.attn_proj_bias,
        )
        self.wv = nn.Linear(
            params.dim,
            params.n_kv_heads * params.head_dim,
            bias=params.attn_proj_bias,
        )
        self.wo = nn.Linear(
            params.n_heads * params.head_dim,
            params.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[DecodingCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.params.n_heads, self.params.head_dim)
        xk = xk.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)
        xv = xv.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)

        xq = self.rope(xq, input_pos=input_pos)
        xk = self.rope(xk, input_pos=input_pos)

        if cache is not None:
            xk, xv = cache.update_kv(self.layer_id, xk, xv)

        if self.params.attention_impl == AttentionImpl.FLASH:
            assert attn_mask is None
            output = _flash_attention(xq, xk, xv)
        elif self.params.attention_impl == AttentionImpl.SDPA:
            output = _sdpa_attention(xq, xk, xv, attn_mask)
        else:
            output = _eager_attention(xq, xk, xv, attn_mask)

        return self.wo(output)

    def forward_scores(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.params.n_heads, self.params.head_dim)
        xk = xk.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)
        xv = xv.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)

        xq = self.rope(xq, input_pos=input_pos)
        xk = self.rope(xk, input_pos=input_pos)

        output, scores = _eager_attention(
            xq,
            xk,
            xv,
            attn_mask,
            output_scores=True,
        )

        return self.wo(output), scores


class FeedForward(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        hidden_dim = int(params.ffn_dim_multiplier * params.dim)
        hidden_dim = params.multiple_of * (
            (hidden_dim + params.multiple_of - 1) // params.multiple_of
        )

        self.w1 = nn.Linear(params.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, params.dim, bias=False)
        self.w3 = nn.Linear(params.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TiedLinear(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = [embedding]

    def forward(self, x):
        return nn.functional.linear(x, self.embedding[0].weight)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.attention = Attention(layer_id, params)
        self.feed_forward = FeedForward(params)
        self.attention_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.ffn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[DecodingCache] = None,
    ):
        h = x + self.attention(
            x=self.attention_norm(x),
            role_ids=role_ids,
            attn_mask=attn_mask,
            input_pos=input_pos,
            cache=cache,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward_scores(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        h, scores = self.attention.forward_scores(
            x=self.attention_norm(x),
            role_ids=role_ids,
            attn_mask=attn_mask,
            input_pos=input_pos,
        )
        h = x + h
        out = h + self.ffn_norm(self.feed_forward(h))
        return out, scores


class Transformer(nn.Module):
    @classmethod
    def from_params(cls, params: ModelParams):
        if params.olmo2_arch:
            from torchllms.models.networks_olmo import OLMo2Transformer
            return OLMo2Transformer(params)
        else:
            return cls(params)

    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        if params.use_role_embeddings:
            self.role_embeddings = RoleEmbeddings(params)
        else:
            self.role_embeddings = None

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        if params.tie_word_embeddings:
            self.output = lambda x: nn.functional.linear(x, self.tok_embeddings.weight)
        else:
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def init_cache(
        self, max_batch_size: int, device: str, max_cache_len: Optional[int] = None
    ):
        return DecodingCache(
            self.params.n_layers,
            max_batch_size,
            max_cache_len or self.params.max_seq_len,
            self.params.n_kv_heads,
            self.params.head_dim,
            device,
            dtype=self.tok_embeddings.weight.dtype,
        )

    def get_wd_params(self):
        wd_params = [
            p
            for n, p in self.named_parameters()
            if p.requires_grad and not _bias_or_norm(n)
        ]
        no_wd_params = [
            p
            for n, p in self.named_parameters()
            if p.requires_grad and _bias_or_norm(n)
        ]
        return wd_params, no_wd_params

    def forward(
        self,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[DecodingCache] = None,
    ):
        """
        Apply a single forward pass for training or inference (prefill + decoding).

        Args:
            input_ids: Per-token input IDs to the model. (bsz, seqlen)
            role_ids: Optional per-token role IDs to the model. (bsz, seqlen)
            attn_mask: Optional attention mask specifying token indices to completely
                ignore. 1 for tokens to attend, 0 for tokens to ignore. (bsz, seqlen)
            input_pos: Optional per-token input positions to the model used for
                positional embeddings. (bsz, seqlen) or (seqlen,)
            cache: Optional decoding cache (past kv, position ids, etc).

        Returns:
            logits: The model's output logits for all input tokens.
            cache: The updated cache, if provided.

        In decoding steps during generation, `input_ids` and `role_ids` should
        contain only the next time step. Batched generation may use `attn_mask` and
        `input_pos` for the first prefill step. Decoding steps will automatically
        extend both if not provided.

        Default behaviors:
        - Missing `role_ids` skips role embeddings.
        - Missing `attn_mask` attends to all tokens autoregressively.
        - Missing `input_pos` applies positional embeddings starting from 0.
        - Missing `cache` skips reading/writing past key/value vectors and role IDs.
        """

        assert (
            cache is None or not cache.is_full()
        ), "Maximum sequence length reached, KV cache is full"

        # RoPE assumes [0, 1, 2, ...] input_pos when not provided
        if input_pos is None and cache is not None:
            input_pos = torch.arange(input_ids.shape[1])[None, :]
            input_pos = input_pos.to(input_ids.device) + cache.next_start_pos[:, None]

        h = self.tok_embeddings(input_ids)

        if self.role_embeddings is not None and role_ids is not None:
            h += self.role_embeddings(role_ids)

        if cache is not None:
            role_ids = cache.update_role_ids(role_ids)
            attn_mask = cache.update_attn_mask(attn_mask)
            cache.next_start_pos = input_pos[:, -1] + 1

        for i, layer in enumerate(self.layers):
            h = layer(h, role_ids, attn_mask, input_pos, cache)

        logits = self.output(self.norm(h))

        return logits, cache

    def forward_scores(
        self,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if attn_mask is not None:
            input_pos = torch.cumsum(attn_mask, dim=1) - 1
        else:
            input_pos = None

        h = self.tok_embeddings(input_ids)

        if self.role_embeddings is not None and role_ids is not None:
            h += self.role_embeddings(role_ids)

        all_scores = []
        for i, layer in enumerate(self.layers):
            h, scores = layer.forward_scores(h, role_ids, attn_mask, input_pos)
            all_scores.append(scores)

        logits = self.output(self.norm(h))

        return logits, all_scores
