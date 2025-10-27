import abc
from typing import Optional

import torch
import torch.nn.functional as F

from torchllms.inference.utils import get_plausibility_mask, logits_to_probs
from torchllms.models.networks import Transformer


class BaseContrastiveLogitsProcessor(abc.ABC):
    def __init__(self, plausibility_threshold: Optional[float] = None):
        self.model = None
        self.cache = None
        self.plausibility_threshold = plausibility_threshold

    @abc.abstractmethod
    def combine_logits(
        self, conditional_logprobs: torch.Tensor, neg_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """How to combine conditional and negative/unconditional log-probs."""
        pass

    def __call__(
        self,
        conditional_logits: torch.Tensor,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            conditional_logits (torch.Tensor): Logits for the conditional branch.
            input_ids (torch.Tensor): Input ids for the negative input.
            role_ids (Optional[torch.Tensor], optional): Role ids for the negative input. Defaults to None.
            attn_mask (Optional[torch.Tensor], optional): Attention mask for the negative input. Defaults to None.
            input_pos (Optional[torch.Tensor], optional): Input positions for the negative input. Defaults to None.

        Returns:
            torch.Tensor: The modified logits. The shape is (bsz, vocab_size).
        """
        assert (
            self.cache is not None and self.model is not None
        ), "Logit processor is not initialized properly"

        conditional_logprobs = F.log_softmax(conditional_logits, dim=-1)

        if self.plausibility_threshold:
            mask = get_plausibility_mask(
                logits_to_probs(conditional_logits), self.plausibility_threshold
            )
            conditional_logprobs[mask] = -float("inf")

        with torch.inference_mode():
            logits, self.cache = self.model(
                input_ids=input_ids,
                role_ids=role_ids,
                cache=self.cache,
                attn_mask=attn_mask,
                input_pos=input_pos,
            )
        neg_logprobs = F.log_softmax(logits[:, -1], dim=-1)

        return self.combine_logits(conditional_logprobs, neg_logprobs)

    def init_model_and_cache(
        self,
        model: Transformer,
        batch_size: int,
        device: str,
        max_cache_len: Optional[int] = None,
    ):
        self.model = model
        self.cache = self.model.init_cache(batch_size, device, max_cache_len)

    def evict_cache(self, evict_mask: torch.Tensor):
        assert self.cache is not None, "Logit processor is not initialized properly"
        self.cache.evict(evict_mask)


class CFGLogitsProcessor(BaseContrastiveLogitsProcessor):
    def __init__(self, guidance_scale: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance_scale = guidance_scale

    def combine_logits(self, conditional_logprobs, neg_logprobs):
        return neg_logprobs + self.guidance_scale * (
            conditional_logprobs - neg_logprobs
        )


class ContrastiveDecodingLogitsProcessor(BaseContrastiveLogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def combine_logits(self, conditional_logprobs, neg_logprobs):
        return conditional_logprobs - neg_logprobs


PROCESSORS = {
    "cd": ContrastiveDecodingLogitsProcessor,
    "cfg": CFGLogitsProcessor,
}
