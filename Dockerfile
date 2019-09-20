# FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
FROM nvcr.io/nvidia/pytorch:19.09-py3

ENV PATH $PATH:/root/.local/bin
RUN apt-get update && apt-get install -y vim

RUN pip install cython && pip install --user \
    comet-ml \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    sktime \
    tslearn \
    tqdm

COPY . /workspace/
