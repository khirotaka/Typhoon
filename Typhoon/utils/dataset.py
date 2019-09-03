import os
import zipfile
import tqdm
import requests
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from .functions import FixedSlidingWindow

SAVE_DIR = os.getcwd()


def fetch_from_url(url: str) -> str:
    filename = url.split("/")[-1]
    file_size = int(requests.head(url).headers["content-length"])

    r = requests.get(url, stream=True)
    pbar = tqdm.tqdm(total=file_size, unit="B", unit_scale=True)

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))

        pbar.close()

    return SAVE_DIR + "/" + filename


def load_data_from_ts_file(path):
    data, label = load_from_tsfile_to_dataframe(path)
    samples, features = data.shape

    tmp = []
    for i in range(samples):
        d_samples = []
        for j in range(features):
            d_samples.append(data.loc[i][j])
        d_samples = np.dstack(d_samples)
        tmp.append(d_samples)

    data = np.vstack(tmp)
    label = pd.get_dummies(label).to_numpy().argmax(1)

    return data, label


def load_har_files(filenames):
    done = []
    for name in filenames:
        data = pd.read_csv(name, header=None, delim_whitespace=True).values
        done.append(data)

    done = np.dstack(done)
    return done


def fetch_har():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    filename = url.split("/")[-1]

    file_size = int(requests.head(url).headers["content-length"])

    r = requests.get(url, stream=True)
    pbar = tqdm.tqdm(total=file_size, unit="B", unit_scale=True)

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))

        pbar.close()

    with zipfile.ZipFile(filename) as zfile:
        zfile.extractall(SAVE_DIR)
    os.remove(filename)


def load_har(raw=True):
    save_dir = SAVE_DIR + "/UCI HAR Dataset/"
    if not os.path.isdir(save_dir):
        print("Downloading UCI HAR Dataset ...")
        fetch_har()

    if raw:
        types = ["train", "test"]
        path_to_raw = "Inertial Signals/"

        def load_raw_data(file_type):
            datasets = [
                "total_acc_x_" + file_type + ".txt", "total_acc_y_" + file_type + ".txt",
                "total_acc_z_" + file_type + ".txt",
                "body_acc_x_" + file_type + ".txt", "body_acc_y_" + file_type + ".txt",
                "body_acc_z_" + file_type + ".txt",
                "body_gyro_x_" + file_type + ".txt", "body_gyro_y_" + file_type + ".txt",
                "body_gyro_z_" + file_type + ".txt"
            ]

            tmp = [save_dir + file_type + "/" + path_to_raw + i for i in datasets]

            X = load_har_files(tmp)
            return X

        raw_datasets = []

        for mode in types:
            data = load_raw_data(mode)
            target = pd.read_csv(save_dir + mode + "/y_" + mode + ".txt", header=None, delim_whitespace=True).values

            raw_datasets.append((data, target))

        (x_train, y_train), (x_test, y_test) = raw_datasets[0], raw_datasets[1]

    else:
        x_train = pd.read_csv(save_dir+"train/X_train.txt", header=None, delim_whitespace=True).values
        y_train = pd.read_csv(save_dir + "train/y_train.txt", header=None, delim_whitespace=True).values

        x_test = pd.read_csv(save_dir + "test/X_test.txt", header=None, delim_whitespace=True).values
        y_test = pd.read_csv(save_dir + "test/y_test.txt", header=None, delim_whitespace=True).values

    return (x_train, y_train), (x_test, y_test)


def fetch_mhealth(extract=True, url=None):
    if not url:
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"

    filename = fetch_from_url(url)

    if extract:
        with zipfile.ZipFile(filename) as zfile:
            zfile.extractall(SAVE_DIR)

    os.remove(filename)


def load_mhealth(items: list, window_size: int, overlap_rate=0.5, drop_null=True) -> (np.ndarray, np.ndarray):
    path = os.getcwd() + "/MHEALTHDATASET/"

    if not os.path.isdir(path):
        fetch_mhealth()

    path = path + "mHealth_subject{}.log"

    data = []
    labels = []
    sw = FixedSlidingWindow(window_size, overlap_rate=overlap_rate)
    for i in items:
        tmp = pd.read_csv(path.format(i), delim_whitespace=True, header=None)
        x = tmp.iloc[:, :-1]
        y = tmp.iloc[:, -1]
        x, y = sw(x, y)
        x = x.astype(np.float32)
        y = y.astype(np.int64)

        if drop_null:
            x = x[y != 0]
            y = y[y != 0]
            y -= 1

        data.append(x)
        labels.append(y)

    data = np.vstack(data)
    labels = np.hstack(labels)

    return data, labels


class MHealth(Dataset):
    def __init__(self, items: list, window_size: int, overlap_rate=0.5, drop_null=True):
        self.data, self.labels = load_mhealth(
            items,
            window_size=window_size,
            overlap_rate=overlap_rate,
            drop_null=drop_null
        )

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        return x, y

    def __len__(self):
        return len(self.labels)


def fetch_epilepsy(extract=True) -> str:
    url = "http://www.timeseriesclassification.com/Downloads/Epilepsy.zip"
    filename = fetch_from_url(url)

    if extract:
        with zipfile.ZipFile(filename) as zf:
            zf.extract("Epilepsy_TRAIN.ts", SAVE_DIR + "/Epilepsy/")
            zf.extract("Epilepsy_TEST.ts", SAVE_DIR + "/Epilepsy/")

    os.remove(filename)
    return SAVE_DIR + "/Epilepsy/"


def load_epilepsy(train=True):
    path = SAVE_DIR + "/Epilepsy/"
    if not os.path.isdir(path):
        fetch_epilepsy()

    path = path + "Epilepsy_{}.ts"
    mode = "TRAIN" if train else "TEST"

    data, label = load_data_from_ts_file(path.format(mode))
    return data, label


def fetch_uwave(extract=True):
    url_x = "http://www.timeseriesclassification.com/Downloads/UWaveGestureLibraryX.zip"
    url_y = "http://www.timeseriesclassification.com/Downloads/UWaveGestureLibraryY.zip"
    url_z = "http://www.timeseriesclassification.com/Downloads/UWaveGestureLibraryZ.zip"
    filename_x = fetch_from_url(url_x)
    filename_y = fetch_from_url(url_y)
    filename_z = fetch_from_url(url_z)

    if extract:
        with zipfile.ZipFile(filename_x) as x:
            x.extract("UWaveGestureLibraryX_TRAIN.ts", SAVE_DIR+"/UWaveGestureLibrary/")
            x.extract("UWaveGestureLibraryX_TEST.ts", SAVE_DIR+"/UWaveGestureLibrary/")

        with zipfile.ZipFile(filename_y) as y:
            y.extract("UWaveGestureLibraryY_TRAIN.ts", SAVE_DIR+"/UWaveGestureLibrary/")
            y.extract("UWaveGestureLibraryY_TEST.ts", SAVE_DIR+"/UWaveGestureLibrary/")

        with zipfile.ZipFile(filename_z) as z:
            z.extract("UWaveGestureLibraryZ_TRAIN.ts", SAVE_DIR+"/UWaveGestureLibrary/")
            z.extract("UWaveGestureLibraryZ_TEST.ts", SAVE_DIR+"/UWaveGestureLibrary/")

    os.remove(filename_x)
    os.remove(filename_y)
    os.remove(filename_z)
    return SAVE_DIR + "/UWaveGestureLibrary/"


def load_uwave(train=True):
    path = SAVE_DIR + "/UWaveGestureLibrary/"
    if not os.path.isdir(path):
        fetch_uwave()

    path = path + "UWaveGestureLibrary{}_{}.ts"
    mode = "TRAIN" if train else "TEST"

    x_data, x_label = load_data_from_ts_file(path.format("X", mode))
    y_data, y_label = load_data_from_ts_file(path.format("Y", mode))
    z_data, z_label = load_data_from_ts_file(path.format("Z", mode))

    data = np.dstack([x_data, y_data, z_data])
    label = pd.get_dummies(x_label).to_numpy().argmax(1)
    return data, label
