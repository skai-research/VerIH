import abc
import math
import re
from contextlib import contextmanager
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_replace_linear_with_lora(rank: int, alpha: int, dropout: float):
    def replace_linear_with_lora(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                module.add_module(
                    name,
                    LoRALinear.from_linear(child, rank, alpha, dropout),
                )

    return replace_linear_with_lora


def replace_lora_with_linear(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            module.add_module(name, child.to_linear())


def make_replace_embedding_with_lora(rank: int, alpha: int, dropout: float):
    def replace_embedding_with_lora(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Embedding):
                module.add_module(
                    name, LoRAEmbedding.from_embedding(child, rank, alpha, dropout)
                )

    return replace_embedding_with_lora


def replace_lora_with_embedding(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, LoRAEmbedding):
            module.add_module(name, child.to_embedding())


def freeze_non_lora_params(module: nn.Module):
    if isinstance(module, LoRALinear):
        module.lora_A.requires_grad = True
        module.lora_B.requires_grad = True
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False
    elif isinstance(module, LoRAEmbedding):
        module.lora_A.requires_grad = True
        module.lora_B.requires_grad = True
        module.weight.requires_grad = False
    else:
        for param in module.parameters(recurse=False):
            param.requires_grad = False


def unfreeze_params(module: nn.Module, params: List[str]):
    for name, param in module.named_parameters():
        if any(re.fullmatch(regex, name) for regex in params):
            param.requires_grad = True


@contextmanager
def disable_lora(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRAModule):
            module.enabled = False

    try:
        yield
    finally:
        for module in model.modules():
            if isinstance(module, LoRAModule):
                module.enabled = True


class LoRAModule(abc.ABC):
    enabled: bool = True


class LoRALinear(nn.Module, LoRAModule):
    """
    Register linear layer weights as buffers so they aren't trainable. FSDP does weird things to buffers but should be
    fine for our use case (https://github.com/pytorch/pytorch/issues/120737).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(
            torch.empty((lora_rank, in_features), **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.empty((out_features, lora_rank), **factory_kwargs)
        )
        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha

        self.reset_parameters()

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, lora_rank: int, lora_alpha: int, lora_dropout: float
    ):
        lora_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        lora_linear.weight.data = linear.weight.data
        if linear.bias is not None:
            lora_linear.bias.data = linear.bias.data
        return lora_linear

    def to_linear(self):
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        linear.weight.data = self.weight.data
        lora_weight = self.lora_B @ self.lora_A * self.lora_alpha / self.lora_rank
        linear.weight.data += lora_weight.to(linear.weight.device)
        if self.bias is not None:
            linear.bias.data = self.bias.data
        return linear

    def reset_parameters(self) -> None:
        """Does not reset underlying linear layer weights!"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.linear(input, self.weight, self.bias)

        if not self.enabled:
            return out

        out_lora = F.linear(F.dropout(input, p=self.lora_dropout), self.lora_A)
        out_lora = F.linear(out_lora, self.lora_B)
        return out + out_lora * self.lora_alpha / self.lora_rank


class LoRAEmbedding(nn.Module, LoRAModule):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
            requires_grad=False,
        )
        self.lora_A = nn.Parameter(
            torch.empty((num_embeddings, lora_rank), **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.empty((lora_rank, embedding_dim), **factory_kwargs)
        )
        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Does not reset underlying embedding layer weights!"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.embedding(input, self.weight)

        if not self.enabled:
            return out

        out_lora = F.embedding(F.dropout(input, p=self.lora_dropout), self.lora_A)
        out_lora = out_lora @ self.lora_B
        return out + out_lora * self.lora_alpha / self.lora_rank

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        lora_embedding = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            device=embedding.weight.device,
            dtype=embedding.weight.dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        lora_embedding.weight.data = embedding.weight.data
        return lora_embedding

    def to_embedding(self):
        embedding = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        embedding.weight.data = self.weight.data
        lora_weight = self.lora_A @ self.lora_B * self.lora_alpha / self.lora_rank
        embedding.weight.data += lora_weight.to(embedding.weight.device)
        return embedding
