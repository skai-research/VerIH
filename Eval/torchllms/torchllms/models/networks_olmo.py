from typing import Optional

import torch
import torch.nn as nn

from torchllms.models.cache import DecodingCache
from torchllms.models.networks import (
    AttentionImpl,
    FeedForward,
    ModelParams,
    RMSNorm,
    RotaryPositionalEmbeddings,
    Transformer,
    _eager_attention,
    _flash_attention,
    _sdpa_attention,
)


class OLMo2Attention(nn.Module):
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

        self.q_norm = RMSNorm(params.n_heads * params.head_dim, eps=params.norm_eps)
        self.k_norm = RMSNorm(params.n_kv_heads * params.head_dim, eps=params.norm_eps)

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

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

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

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

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


class OLMo2TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.attention = OLMo2Attention(layer_id, params)
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
        h = x + self.attention_norm(
            self.attention(
                x,
                role_ids,
                input_pos,
                cache,
                attn_mask,
            )
        )
        out = h + self.ffn_norm(self.feed_forward(h))
        return out

    def forward_scores(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        h, scores = self.attention.forward_scores(
            x,
            role_ids,
            attn_mask,
            input_pos,
        )
        h = x + self.attention_norm(h)
        out = h + self.ffn_norm(self.feed_forward(h))
        return out, scores


class OLMo2Transformer(Transformer):
    def __init__(self, params: ModelParams):
        super().__init__(params)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(OLMo2TransformerBlock(layer_id, params))
