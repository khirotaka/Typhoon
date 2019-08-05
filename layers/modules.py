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
