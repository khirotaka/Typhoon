from copy import deepcopy
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn as cudnn

from Typhoon.utils.dataset import MHealth
from Typhoon.core.model import TyphoonClassifier
from Typhoon.utils.trainer import NeuralNetworkClassifier

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


in_feature = 23
seq_len = 256
n_heads = 32
factor = 32
num_class = 12

experiment = Experiment()

clf = NeuralNetworkClassifier(
    TyphoonClassifier(in_feature, seq_len, n_heads, factor, num_class, num_layers=6, d_model=256, dropout_rate=0.5),
    nn.CrossEntropyLoss(),
    optim.Adam, {"lr": 0.000001, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4}, experiment
)

init_checkpoints = deepcopy(clf.save_checkpoint())

data = [i for i in range(1, 11, 1)]
tests = [[i, i+1] for i in reversed(range(1, 10, 2))]
vals = [i[0]-1 if i[0] != 1 else 10 for i in tests]

trains = [[i for i in data if i not in tests[j]] for j in range(5)]
_ = [trains[i].remove(j) for i, j in enumerate(vals)]


for k in range(5):
    print("Phase{}".format(k+1))
    clf.restore_checkpoint(init_checkpoints)

    train_ds = MHealth(trains[k], 256)
    val_ds = MHealth([vals[k]], 256)
    test_ds = MHealth(tests[k], 256)

    train_loader = DataLoader(train_ds, 128, False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    clf.experiment_tag = "mhealth/phase{}".format(k+1)
    clf.fit(
        {"train": train_loader,
         "val": val_loader},
        epochs=200
    )
    acc = clf.evaluate(test_loader, True)
    experiment.log_metric("phase{}/accuracy".format(k+1), acc)
    clf.save_to_file("save_params_mhealth/".format(k))
