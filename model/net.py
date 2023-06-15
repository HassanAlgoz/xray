"""Defines the neural network, losss function and metrics"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder


class Net(nn.Module):
    """
    The documentation for all the various components available to you is here:
    http://pytorch.org/docs/master/nn.html
    """

    def __init__(
        self,
        *,
        device: torch.device,
        embeddings: torch.FloatTensor,
        input_window_size: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1
    ):
        super(Net, self).__init__()

        d_model = embeddings.shape[1]
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)

        # Positional Encodings
        pos_enc = torch.zeros(1, input_window_size, d_model)
        for pos in range(input_window_size):
            for i in range(0, d_model, 2):
                pos_enc[0, pos, i] = math.sin(pos / (10000 ** ((i) / d_model)))
                pos_enc[0, pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / d_model)))
        self.positional = pos_enc.to(device)

        self.encoders = nn.ModuleList(
            [TransformerEncoder(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # batch_size, seq_len, d_model = x.shape
        x = self.embedding(x)

        x += self.positional[:, :]

        for enc in self.encoders:
            x = enc(x)

        # Classification (log_softmax is numerically more stable than softmax)
        y_pred = F.softmax(self.fc(x), dim=1)
        return y_pred[:, -1, :]
        # TODO: maybe return ths y_pred as is? (the example does)
        # return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags


loss_fn = torch.nn.CrossEntropyLoss()


def accuracy(outputs, labels):
    y_pred = np.argmax(outputs, axis=1)
    sum_correct = (y_pred == labels).sum().item()
    num_total = labels.shape[0]
    return sum_correct / num_total


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    "accuracy": accuracy,
    # could add more metrics such as accuracy for each token type
}
