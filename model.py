import torch
import torch.nn as nn

from Typhoon import modules


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, num_layers: int) -> None:
        super(EncoderLayer, self).__init__()
        self.layers = nn.ModuleList([modules.EncoderBlock(embed_dim, num_head) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x
