import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float) -> None:
        super(PointWiseFeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.network(tensor)
        return tensor


class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1) -> None:
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.layer, nn.MultiheadAttention):
            output, self.attn_weights = self.layer(x, x, x)

        else:
            output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_head: int) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = ResidualBlock(nn.MultiheadAttention(embed_dim, num_head), embed_dim)
        self.ffn = ResidualBlock(PointWiseFeedForward(embed_dim, dropout_rate=0.1), embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ffn(x)
        return x
