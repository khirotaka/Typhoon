from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split

from Typhoon.utils.dataset import load_har
from Typhoon.core.model import Typhoon
from Typhoon.utils.trainer import NeuralNetworkClassifier

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

(x_train, y_train), (x_test, y_test) = load_har(True)

y_train -= 1
y_test -= 1

y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = torch.tensor(x_train).float()
x_test = torch.tensor(x_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

train_val_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

n_samples = len(train_val_ds)
train_size = int(n_samples * 0.7)

train_idx = list(range(0, train_size))
val_idx = list(range(train_size, n_samples))

train_ds = Subset(train_val_ds, train_idx)
val_ds = Subset(train_val_ds, val_idx)


train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

in_feature = 9
seq_len = 128
n_heads = 32
factor = 32
num_class = 6


clf = NeuralNetworkClassifier(
    Typhoon(in_feature, seq_len, n_heads, factor, num_class, num_layers=6, d_model=128, dropout_rate=0.2),
    nn.CrossEntropyLoss(),
    optim.Adam, {"lr": 0.000001, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4}, Experiment()
)

clf.experiment_tag = "har_dataset"
clf.num_class = num_class

clf.fit(
    {"train": train_loader,
     "val": val_loader},
    epochs=200
)
clf.evaluate(test_loader)
clf.confusion_matrix(test_ds)
clf.save_to_file("save_params_test/")
