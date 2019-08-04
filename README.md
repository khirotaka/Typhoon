# Typhoon
Transformer based neural network model for time series tasks.

## Utilities

```python
from Typhoon.utils.trainer import NeuralNetworkClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(4, 1)
        self.fc2 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)

train_loader = DataLoader(...)

optim_confg = {"lr": 0.01}
comet_config = {"api_key": "YOUR-API-KEY", "project_name": "YOUR-PROJECT_NAME"}

clf = NeuralNetworkClassifier(Network(), nn.BCELoss(), optim.SGD, optim_confg, comet_config)
clf.fit(train_loader, epochs=20)

```

## References 
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
* [Attend and Diagnose: Clinical Time Series Analysis Using Attention Models](https://arxiv.org/abs/1711.03905)
* [TimeNet: Pre-trained deep recurrent neural network for time series classification](https://arxiv.org/abs/1706.08838)
