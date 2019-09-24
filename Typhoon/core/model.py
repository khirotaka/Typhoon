from typing import List
import torch
import torch.nn as nn
from Typhoon.core import modules
from Typhoon.utils.functions import subsequent_mask, silu, positional_encoding, dense_interpolation


class EncoderLayer(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.transform = nn.Linear(input_features, d_model)
        self.positional_enc = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.transform, nn.RNNBase):
            x, h = self.transform(x)        # [N, seq_len, n_directions * hidden_size]
        elif isinstance(self.transform, nn.Conv1d):
            x = x.transpose(1, 2)           # [N, seq_len, features] -> [N, features, seq_len]
            x = self.transform(x)
            x = x.transpose(1, 2)           # [N, seq_len, features]
            x = silu(x)                     # TODO dropout ?
        else:
            x = self.transform(x)           # [N, seq_len, d_model]

        x = self.positional_enc(x)          # [N, seq_len, d_model]

        for l in self.blocks:
            x = l(x)                        # [N, seq_len, d_model]

        return x


class EncoderLayerForSAnD(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2):
        super(EncoderLayerForSAnD, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, target_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model

        self.transform = nn.Linear(target_features, d_model)
        self.positional_enc = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.DecoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

        self.fc = nn.Linear(d_model, target_features)

    def forward(self, src, tgt, mask):
        tgt = self.transform(tgt)
        tgt += self.positional_enc(tgt)

        for l in self.blocks:
            tgt = l(tgt, src, mask)
            tgt = tgt.transpose(0, 1)

        x = self.fc(tgt)
        return x


class TyphoonClassifier(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, factor, num_class, num_layers, d_model=128, dropout_rate=0.2):
        super(TyphoonClassifier, self).__init__()
        self.encoder = EncoderLayer(input_features, seq_len, n_heads, num_layers, d_model, dropout_rate)
        self.denseint = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, num_class)

    def forward(self, x):
        x = self.encoder(x)
        x = self.denseint(x)                # [N, d_model, factor]
        x = self.clf(x)

        return x


class TyphoonAutoEncoder(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2):
        super(TyphoonAutoEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = EncoderLayer(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.decoder = DecoderLayer(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)

    def forward(self, x):
        size = x.shape[1]
        mask = subsequent_mask(size).to(self.device)
        out = self.encoder(x)
        out = self.decoder(out, x, mask)
        return out


class Typhoon(nn.Module):
    def __init__(self, in_sizes: list, seq_len: int, n_heads: List[int], n_layers: List[int], factor: int, n_class: int, d_models: List[int], dropout: float = 0.2) -> None:
        super(Typhoon, self).__init__()
        self.all_dim = sum(d_models)

        self.sub_modules = nn.ModuleList([
            EncoderLayer(dim, seq_len, n_heads[idx], n_layers[idx], d_models[idx], dropout) for idx, dim in enumerate(in_sizes)
        ])
        self.conv = nn.Conv1d(self.all_dim, self.all_dim, 1)
        self.denseint = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(self.all_dim, factor, n_class)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        out = []
        for idx, module in enumerate(self.sub_modules):
            out.append(module(x[idx]))

        out = torch.cat(out, 2)             # [N, seq_len, sum(d_models)]
        out = out.transpose(1, 2)
        out = self.conv(out).transpose(1, 2)
        out = self.denseint(out)
        out = self.clf(out)
        return out


class SAnD(nn.Module):
    """
    Simply Attend and Diagnose model

    The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)

    `Attend and Diagnose: Clinical Time Series Analysis Using Attention Models <https://arxiv.org/abs/1711.03905>`_
    Huan Song, Deepta Rajan, Jayaraman J. Thiagarajan, Andreas Spanias
    """
    def __init__(
            self, input_features: int, seq_len: int, n_heads: int, factor: int,
            n_class: int, n_layers: int, d_model: int = 128, dropout_rate: float = 0.2
    ):
        super(SAnD, self).__init__()
        self.encoder = EncoderLayerForSAnD(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.dense_interpolation(x)
        x = self.clf(x)
        return x


class TyphoonForONNX(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, factor, n_classes, n_layers, d_model=128, dropout_rate=0.2):
        super(TyphoonForONNX, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.factor = factor
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = nn.Linear(input_features, d_model)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])
        self.clf = modules.ClassificationModule(d_model, factor, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = int(x.shape[0])
        di = dense_interpolation(batch_size, self.seq_len, self.factor).to(self.device)
        pe = positional_encoding(self.seq_len, self.d_model).to(self.device)

        x = self.transform(x)
        x = x + pe
        for l in self.blocks:
            x = l(x)
        x = torch.bmm(di, x)
        x = self.clf(x)
        return x
