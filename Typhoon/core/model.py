import torch
import torch.nn as nn
from Typhoon.core import modules
from Typhoon.utils.functions import subsequent_mask


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
        else:
            x = self.transform(x)           # [N, seq_len, d_model]

        x = self.positional_enc(x)         # [N, seq_len, d_model]

        for l in self.blocks:
            x = l(x)                        # [N, seq_len, d_model]

        return x


class EncoderLayerWithLSTM(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2, bidirectional=False):
        super(EncoderLayerWithLSTM, self).__init__()
        flag = d_model // 2 if bidirectional else d_model

        self.d_model = d_model

        self.transform = nn.LSTM(
            input_features,
            flag,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional
        )
        self.positional_enc = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h = self.transform(x)
        out = self.positional_enc(out)

        for l in self.blocks:
            out = l(out)
        return out


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


class TyphoonClassifierWithLSTM(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, factor, num_class, num_layers, d_model=128, dropout_rate=0.2, bidirectional=False):
        super(TyphoonClassifierWithLSTM, self).__init__()
        self.encoder = EncoderLayerWithLSTM(input_features, seq_len, n_heads, num_layers, d_model, dropout_rate, bidirectional)
        self.denseint = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, num_class)

    def forward(self, x):
        x = self.encoder(x)
        x = self.denseint(x)
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
    def __init__(self, input_features, seq_len, n_heads, n_layers, factor, n_class, d_model=128, dropout_rate=0.2):
        super(Typhoon, self).__init__()
        self.ae = TyphoonAutoEncoder(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.denseint = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, x):
        x = self.ae.encoder(x)
        x = self.denseint(x)
        x = self.clf(x)

        return x
