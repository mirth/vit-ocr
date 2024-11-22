import torch
from torch import nn
from einops import rearrange
from attn import CrossAttention

from tokenizer import chars

VOCAB_SIZE = len(chars)
# VOCAB_SIZE = 50257 + 1
# def count_parameters(model):
#     for name, p in model.named_parameters():
#         print(name, p.numel())

class VitEncoder(nn.Module):
    def __init__(self, vit, max_length):
        super().__init__()
        
        self.max_length = max_length
        self.vit = vit
        self.cross_attn = CrossAttention(
            kv_dim=768,
            q_dim=VOCAB_SIZE,
            qk_out_dim=768,
            v_out_dim=768,
            num_heads=768 // 128,
        )
        self.latents = nn.Parameter(torch.randn(max_length, VOCAB_SIZE))
        self.pos_emb = nn.Embedding(197, 768)

    def forward(self, x):
        n, c, h, w = x.shape

        x = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        x = self.cross_attn(inputs_kv=x, inputs_q=self.latents.repeat(n, 1, 1))


        return x
    

