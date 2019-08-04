import sys
from comet_ml import Experiment

import torch
import termcolor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class NeuralNetworkClassifier:
    """
    NeuralNetworkClassifier depend on Comet-ML (https://www.comet.ml/).
    You have to create a project on your workspace of Comet, if you use this class.

    How to use.

    1st, Write your code.
    ----------------------------------------------------------
    # code.py
    from Typhoon.utils.trainer import NeuralNetworkClassifier

    import torch

    class Network(torch.nn.Module):
        def __init__(self):
            super(Network ,self).__init__()
            ...
        def forward(self, x):
            ...

    optimizer_config = {"lr": 0.001, "betas": , "eps"}

    clf = NeuralNetworkClassifier(
            Network(), nn.CrossEntropyLoss(),
            optim.Adam, optimizer_config
        )
    clf.fit(train_loader, epochs=10)
    clf.evaluate(test_loader)
    ----------------------------------------------------------

    2nd, Run code on your shell.
    > export COMET_API_KEY="YOUR-API-KEY"
    > export COMET_PROJECT_NAME="YOUR-PROJECT-NAME"
    > user@user$ python code.py

    3rd, check logs on your workspace of comet.

    ----------------------------------------------------------

    ====================== EXPERIMENTAL ======================
    Warning, This option is now experimental.

    If your want to check graph of your model,
    setting a logs directory and assign when you define NeuralNetworkClassifier.

    ----------------------------------------------------------
    from Typhoon.utils.trainer import NeuralNetworkClassifier

    clf = NeuralNetworkClassifier(
            Network(), nn.CrossEntropyLoss(),
            optim.Adam, optimizer_config,
            log_dir='logs/'
        )

    clf.fit(train_loader, epochs=10)
    ----------------------------------------------------------

    After that, run TensorBoard and check your model.

    > user@user$ tensorboard --logdir=logs/

    """
    def __init__(self, model, criterion, optimizer, optimizer_config: dict, comet_config: dict, log_dir=None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer(model.parameters(), **optimizer_config)
        self.criterion = criterion

        self.hyper_params = optimizer_config
        self.experiment = Experiment(**comet_config)

        self._tb = False
        self._is_parallel = False

        if isinstance(log_dir, str):
            self.writer = SummaryWriter(log_dir=log_dir)
            self._tb = True

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self._is_parallel = True

            notice = "Running on {} GPUs.".format(torch.cuda.device_count())
            notice = termcolor.colored(notice, "green")
            print(notice)

    def fit(self, loader: DataLoader, epochs: int) -> None:
        """
        The method of training your PyTorch Model.
        With the assumption, This method use for training network for classification.
        This is automatically logging a Network Graph using TensorBoard, Hyper Parameters, Losses, and Accuracy.

        ---------------------------------------------------------
        clf = NeuralNetworkClassifier(
                Network(), nn.CrossEntropyLoss(),
                optim.Adam, optimizer_config
            )
        clf.fit(train_loader, epochs=10)
        ---------------------------------------------------------

        :param loader: DataLoader for Training  : torch.utils.data.DataLoader
        :param epochs: The number of epochs: int
        :return: None
        """
        batch_size = loader.batch_size
        len_of_dataset = len(loader.dataset)
        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = batch_size
        self.experiment.log_parameters(self.hyper_params)

        self.model.train()

        with self.experiment.train():
            for epoch in range(epochs):
                correct = 0.0
                total = 0.0
                for batch, (x, y) in enumerate(loader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    if self._tb and (batch == 0 and epoch == 0):
                        self.writer.add_graph(self.model, x)

                    sys.stdout.write(
                        "\rTraining - Epochs: {:03d}/{:03d} - {:.3%} ".format(
                            epoch + 1, epochs, ((batch_size * (batch + 1)) / len_of_dataset)
                        )
                    )
                    sys.stdout.flush()

                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    total += y.shape[0]
                    correct += (predicted == y).sum().float().item()

                    self.experiment.log_metric("loss", loss.item(), step=batch)
                    self.experiment.log_metric("accuracy", float(correct / total), step=batch)

                    loss.backward()
                    self.optimizer.step()

            sys.stdout.write("\n")

        if self._tb:
            self.writer.close()

    def evaluate(self, loader: DataLoader) -> None:
        """
        The method of evaluating your PyTorch Model.
        With the assumption, This method use for training network for classification.
        ---------------------------------------------------------
        clf = NeuralNetworkClassifier(
                Network(), nn.CrossEntropyLoss(),
                optim.Adam, optimizer_config
            )
        clf.evaluate(test_loader)
        ---------------------------------------------------------

        :param loader: DataLoader for Evaluating: torch.utils.data.DataLoader
        :return: None
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0.0

        with self.experiment.test():
            with torch.no_grad():
                correct = 0
                total = 0
                for batch, (x, y) in enumerate(loader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    total += y.shape[0]
                    correct += (predicted == y).sum()

                    running_loss += loss.item()
                    running_corrects += torch.sum(predicted == y).item()

                    self.experiment.log_metric("loss", running_loss, step=batch)
                    self.experiment.log_metric("accuracy", float(running_corrects / total), step=batch)

        print("Evaluate is finished. See your workspace https://www.comet.ml/")

    def save_weights(self, path: str) -> None:
        """
        The method of saving trained PyTorch model.

        ---------------------------------------------------------
        clf = NeuralNetworkClassifier(
                Network(), nn.CrossEntropyLoss(),
                optim.Adam, optimizer_config
            )
        clf.fit(train_loader, epochs=10)
        clf.save_weights('path/to/save/dir/and/filename.pth')
        ---------------------------------------------------------

        :param path: path to save directory. : str
        :return: None
        """
        if self._is_parallel:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_weight(self, path: str) -> None:
        """
        The method of loading trainer PyTorch model.

        ---------------------------------------------------------
        clf = NeuralNetworkClassifier(
                Network(), nn.CrossEntropyLoss(),
                optim.Adam, optimizer_config
            )
        clf.load_weight('path/to/trained/weights.pth')
        ---------------------------------------------------------

        :param path: path to saved directory. : str
        :return: None
        """
        map_location = None if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(torch.load(path, map_location=map_location))
