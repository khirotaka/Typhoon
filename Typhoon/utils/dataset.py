import os
import zipfile
import tqdm
import requests
import numpy as np
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe, load_from_arff_to_dataframe

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


def fetch_mhealth(extract=True, url=None):
    if not url:
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"

    filename = fetch_from_url(url)

    if extract:
        with zipfile.ZipFile(filename) as zfile:
            zfile.extractall(SAVE_DIR)

    os.remove(filename)


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
