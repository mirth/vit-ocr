import math
import torch
from torch import nn
from attn import CrossAttention

VOCAB_SIZE = 50257 + 1
# def count_parameters(model):
#     for name, p in model.named_parameters():
#         print(name, p.numel())

class VitEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()

        vit.heads = nn.Identity()
        self.vit = vit
        self.cross_attn = CrossAttention(
            kv_dim=768,
            q_dim=VOCAB_SIZE,
            qk_out_dim=768,
            v_out_dim=768,
            num_heads=768 // 128,
        )
        self.latents = nn.Parameter(torch.randn(10, VOCAB_SIZE))
        self.latents_pos_embeddings = SinusoidalPositionalEmbedding(embed_dim=768, max_seq_length=197)

    def forward(self, x):
        n, c, h, w = x.shape

        x = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)
        x = self.latents_pos_embeddings(x)
        x = self.cross_attn(inputs_kv=x, inputs_q=self.latents.repeat(n, 1, 1))


        return x
    

class SinusoidalPositionalEmbedding(nn.Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x