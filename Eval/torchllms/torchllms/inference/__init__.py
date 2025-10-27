from . import providers, utils
from .llm import LLM
from .llm_cfg import ContrastiveLLM

__all__ = ["providers", "utils", "LLM", "ContrastiveLLM"]

def set_seed(seed):
    import torch
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)