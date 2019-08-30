from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn as cudnn
from torch.utils.data import DataLoader

from Typhoon.utils.dataset import MHealth
from Typhoon.core.model import Typhoon
from Typhoon.utils.trainer import NeuralNetworkClassifier

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_ds = MHealth([1, 2, 3, 4, 5, 6, 7, 8], 256)
val_ds = MHealth([9], 256)
test_ds = MHealth([10], 256)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

in_feature = 23
seq_len = 256
n_heads = 32
factor = 32
num_class = 12


clf = NeuralNetworkClassifier(
    Typhoon(in_feature, seq_len, n_heads, factor, num_class, num_layers=6, d_model=256, dropout_rate=0.5),
    nn.CrossEntropyLoss(),
    optim.Adam, {"lr": 0.000001, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4}, Experiment()
)

clf.experiment_tag = "mhealth_dataset"
clf.num_class = num_class

clf.fit(
    {"train": train_loader,
     "val": val_loader},
    epochs=200
)
clf.evaluate(test_loader)
clf.confusion_matrix(test_ds)
clf.save_to_file("save_params_test/")
