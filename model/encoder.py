import torch
from torch import nn
import torch.nn.functional as F

from .layer_norm import LayerNorm
from .attention_head import AttentionHead


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(TransformerEncoder, self).__init__()

        # Multi-headed Attention
        dh = d_model // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, dh) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * d_model, d_model)
        self.ln1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        # Feed Forward
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.ln2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # batch_size, seq_len, d_model = x.shape

        # Multi-head Attention -> Add & Norm
        multi_head = self.linear(torch.cat([h(x) for h in self.heads], dim=-1))
        x = self.drop1(self.ln1(x + multi_head))

        # Feed Forward -> Add & Norm
        ffn = self.ff2(F.relu(self.ff1(x)))
        x = self.drop2(self.ln2(x + ffn))

        return x
