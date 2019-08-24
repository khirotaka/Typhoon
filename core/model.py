import torch
import torch.nn as nn

from Typhoon.core import modules


class EncoderLayer(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, factor, num_layers, d_model=128, dropout_rate=0.2):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.factor = factor

        self.transform = nn.Linear(input_features, d_model)
        self.positional_enc = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(num_layers)
        ])
        self.denseint = modules.DenseInterpolation(seq_len, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)               # [N, seq_len, d_model]
        x += self.positional_enc(x)         # [N, seq_len, d_model]

        for l in self.blocks:
            x = l(x)                        # [N, seq_len, d_model]

        x = self.denseint(x)                # [N, d_model, factor]

        return x


class Typhoon(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, factor, num_class, num_layers, d_model=512, dropout_rate=0.1):
        super(Typhoon, self).__init__()
        self.encoder = EncoderLayer(input_features, seq_len, n_heads, factor, num_layers, d_model, dropout_rate)
        self.cls = modules.ClassificationModule(d_model, factor, num_class)

    def forward(self, x):
        x = self.encoder(x)
        x = self.cls(x)
        return x
