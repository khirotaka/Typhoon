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
    comet_config = {}

    clf = NeuralNetworkClassifier(
            Network(), nn.CrossEntropyLoss(),
            optim.Adam, optimizer_config, comet_config
        )
    clf.fit(train_val_loader, epochs=10)
    clf.evaluate(test_loader)
    ----------------------------------------------------------

    2nd, Run code on your shell.
    > export COMET_API_KEY="YOUR-API-KEY"
    > export COMET_PROJECT_NAME="YOUR-PROJECT-NAME"
    > user@user$ python code.py

    3rd, check logs on your workspace of comet.

    Note,
    Execute this command on your shell,

    > export COMET_DISABLE_AUTO_LOGGING=1

    If the following error occurs.

    ImportError: You must import Comet before these modules: torch

    ----------------------------------------------------------

    """
    def __init__(self, model, criterion, optimizer, optimizer_config: dict, comet_config: dict) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer(model.parameters(), **optimizer_config)
        self.criterion = criterion
        self.experiment = Experiment(**comet_config)

        optimizer_config["optimizer"] = optimizer
        optimizer_config["criterion"] = criterion

        self.hyper_params = optimizer_config
        self.__num_classes = None

        self._is_parallel = False

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self._is_parallel = True

            notice = "Running on {} GPUs.".format(torch.cuda.device_count())
            print("\033[33m" + notice + "\033[0m")

    def fit(self, loader: dict, epochs: int) -> None:
        """
        The method of training your PyTorch Model.
        With the assumption, This method use for training network for classification.

        ---------------------------------------------------------
        train_ds = Subset(train_val_ds, train_index)
        val_ds = Subset(train_val_ds, val_index)

        train_val_loader = {
            "train": DataLoader(train_ds, batch_size),
            "val": DataLoader(val_ds, batch_size)
        }

        clf = NeuralNetworkClassifier(
                Network(), nn.CrossEntropyLoss(),
                optim.Adam, optimizer_config
            )
        clf.fit(train_val_loader, epochs=10)
        ---------------------------------------------------------

        :param loader: Dictionary which contains Data Loaders for training and validation.: dict{DataLoader, DataLoader}
        :param epochs: The number of epochs: int
        :return: None
        """
        len_of_train_dataset = len(loader["train"].dataset)
        len_of_val_dataset = len(loader["val"].dataset)

        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset
        self.hyper_params["val_ds_size"] = len_of_val_dataset
        self.experiment.log_parameters(self.hyper_params)

        for epoch in range(epochs):
            correct = 0.0
            total = 0.0

            with self.experiment.train():
                self.model.train()
                pbar = tqdm.tqdm(total=len_of_train_dataset)
                for x, y in loader["train"]:
                    b_size = x.shape[0]
                    total += y.shape[0]
                    x = x.to(self.device)
                    y = y.to(self.device)

                    pbar.set_description(
                        "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                    )
                    pbar.update(b_size)

                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().float().cpu().item()

                    self.experiment.log_metric("loss", loss.cpu().item(), step=epoch)
                    self.experiment.log_metric("accuracy", float(correct / total), step=epoch)

                with self.experiment.validate():
                    self.model.eval()
                    val_correct = 0.0
                    val_total = 0.0

                    for x_val, y_val in loader["val"]:
                        val_total += y_val.shape[0]
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        val_output = self.model(x_val)
                        val_loss = self.criterion(val_output, y_val)
                        _, val_pred = torch.max(val_output, 1)
                        val_correct += (val_pred == y_val).sum().float().cpu().item()

                        self.experiment.log_metric("loss", val_loss.cpu().item(), step=epoch)
                        self.experiment.log_metric("accuracy", float(val_correct / val_total), step=epoch)

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
        self.experiment.log_parameter("test_ds_size", len(loader.dataset))

        with self.experiment.test():
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                for step, (x, y) in enumerate(loader):
                    b_size = x.shape[0]
                    total += y.shape[0]
                    x = x.to(self.device)
                    y = y.to(self.device)

                    pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
                    pbar.update(b_size)

                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().float().cpu().item()

                    running_loss += loss.cpu().item()
                    running_corrects += torch.sum(predicted == y).float().cpu().item()

                    self.experiment.log_metric("loss", running_loss, step=step)
                    self.experiment.log_metric("accuracy", float(running_corrects / total), step=step)
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

    @property
    def num_classes(self) -> int or None:
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, num_class: int) -> None:
        assert isinstance(num_class, int) and num_class > 0, "the number of classes must be greater than 0."
        self.__num_classes = num_class
        self.experiment.log_parameter("classes", self.__num_classes)
