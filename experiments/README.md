# Experiments
How to execute the each experiment codes.

## Execution environment
* OS: Unix-like
    * macOS Mojave 10.14.6
    * Debian 9.9 (GCP Deep Learning Image: PyTorch 1.1.0 and fastai m32)
    * Ubuntu 16.04 or Ubuntu 18.04
    
* Python 3.5 or later
* PyTorch 1.1.0

* (optional) NVIDIA GPU


## Execute code
```shell script
root@root:/Typhoon# pwd                                   # ~/Typhoon/Typhoon
root@root:/Typhoon# mv experiments/train_***.py ../
root@root:/Typhoon# export COMET_API_KEY="YOUR-API-KEY"
root@root:/Typhoon# export COMET_PROJECT_NAME="YOUR-PROJECT-NAME"
root@root:/Typhoon# python train_***.py
```


## Docker
