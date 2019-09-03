# Typhoon
Transformer based neural network model for time series tasks.

## Structure

```
.
├── Dockerfile
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── docker-compose.yml
│
├── Typhoon
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── modules.py
│   │
│   └── utils
│       ├── __init__.py
│       ├── dataset.py
│       ├── functions.py
│       └── trainer.py
│
└── experiments
    ├── README.md
    ├── check.py
    ├── train_har.py
    └── train_mhealth.py

```
* `utils/` ... This directory contain training script and sub functions.

## System Requirements
* OS: Unix-like
    * macOS Mojave 10.14.6
    * Debian 9.9 (GCP Deep Learning Image: PyTorch 1.1.0 and fastai m32)
    * Ubuntu 16.04 or Ubuntu 18.04
    
* Python 3.5 or later
* PyTorch 1.1.0

* (optional) NVIDIA GPU

## References 
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
* [Attend and Diagnose: Clinical Time Series Analysis Using Attention Models](https://arxiv.org/abs/1711.03905)
* [TimeNet: Pre-trained deep recurrent neural network for time series classification](https://arxiv.org/abs/1706.08838)
