import math

import torch
from torch import nn


class AttentionHead(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, d_model, dk):
        super(AttentionHead, self).__init__()

        # Attention layer
        self.w_q = nn.Linear(d_model, dk)
        self.w_k = nn.Linear(d_model, dk)
        self.w_v = nn.Linear(d_model, dk)
        self.softmax = nn.Softmax(dim=-1)

        # Feed Forward layer
        self.fc = nn.Linear(dk, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Apply linear transformations
        q = self.w_q(x).view(batch_size, seq_len, -1)
        k = self.w_k(x).view(batch_size, -1, seq_len)
        v = self.w_v(x)

        # Calculate context
        attention_scores = torch.bmm(q, k) / math.sqrt(d_model)
        attention_weights = self.softmax(attention_scores)
        context = torch.bmm(attention_weights, v)

        output = self.softmax(self.fc(context))

        return output
