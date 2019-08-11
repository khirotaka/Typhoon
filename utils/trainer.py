import sys
from comet_ml import Experiment

import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

    """
    def __init__(self, model, criterion, optimizer, optimizer_config: dict, comet_config: dict) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer(model.parameters(), **optimizer_config)
        self.criterion = criterion

        self.hyper_params = optimizer_config
        self.experiment = Experiment(**comet_config)

        self._is_parallel = False

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self._is_parallel = True

            notice = "Running on {} GPUs.".format(torch.cuda.device_count())
            print("\033[33m" + notice + "\033[0m")

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

                pbar = tqdm.tqdm(total=len_of_dataset)
                for batch, (x, y) in enumerate(loader):
                    b_size = x.shape[0]
                    x = x.to(self.device)
                    y = y.to(self.device)

                    pbar.set_description(
                        "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                    )
                    pbar.update(b_size)

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
                pbar.close()

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
        pbar = tqdm.tqdm(total=len(loader.dataset))

        with self.experiment.test():
            with torch.no_grad():
                correct = 0
                total = 0
                for batch, (x, y) in enumerate(loader):
                    b_size = x.shape[0]
                    x = x.to(self.device)
                    y = y.to(self.device)

                    pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
                    pbar.update(b_size)

                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    total += y.shape[0]
                    correct += (predicted == y).sum()

                    running_loss += loss.item()
                    running_corrects += torch.sum(predicted == y).item()

                    self.experiment.log_metric("loss", running_loss, step=batch)
                    self.experiment.log_metric("accuracy", float(running_corrects / total), step=batch)
                pbar.close()

        print("\033[33m" + "="*65 + "\033[0m")
        print("\033[33m" + "Evaluation finished. Check your workspace" + "\033[0m" + " https://www.comet.ml/")
        print("\033[33m" + "="*65 + "\033[0m")

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
