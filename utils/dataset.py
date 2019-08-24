import os
import zipfile
import requests
import tqdm
import numpy as np
import pandas as pd

SAVE_DIR = os.getcwd()


def fetch_mhealth(extract=True, url=None):
    if not url:
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"

    filename = url.split("/")[-1]
    file_size = int(requests.head(url).headers["content-length"])

    r = requests.get(url, stream=True)
    pbar = tqdm.tqdm(total=file_size, unit="B", unit_scale=True)

    print("Downloading UCI MHEALTH dataset ...")
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))

        pbar.close()

    if extract:
        with zipfile.ZipFile(filename) as zfile:
            zfile.extractall(SAVE_DIR)

    os.remove(filename)
