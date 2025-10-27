import torch
import torch.nn as nn

from torchllms.models import lora

x = torch.randn(1, 10)
linear = nn.Linear(10, 12)
lora_linear = lora.LoRALinear.from_linear(
    linear, lora_rank=8, lora_alpha=16, lora_dropout=0
)

print("linear(x)")
print(linear(x))
print("lora_linear(x)")
print(lora_linear(x))

linear_from_lora = lora.LoRALinear.to_linear(lora_linear)
print("linear_from_lora(x)")
print(linear_from_lora(x))

x = torch.randint(0, 10, (1, 10))
embedding = nn.Embedding(10, 12)
lora_embedding = lora.LoRAEmbedding.from_embedding(
    embedding, lora_rank=8, lora_alpha=16, lora_dropout=0
)
print("embedding(x)")
print(embedding(x))
print("lora_embedding(x)")
print(lora_embedding(x))

embedding_from_lora = lora.LoRAEmbedding.to_embedding(lora_embedding)
print("embedding_from_lora(x)")
print(embedding_from_lora(x))
