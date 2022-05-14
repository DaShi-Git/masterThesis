# Master's Thesis: Automatic Code Generation for Kernel Fusion
## Environment
Required environment:
 - NVIDIA GPU with RTX, e.g. RTX20xx or RTX30xx (we use an RTX2070)
 - CUDA 11
 - Python 3.8 or higher, see `environment.txt` for the required packages

Tested systems:
- Ubuntu 20.04, gcc 9.4.0, CUDA 11.5, Python 3.8, PyTorch 1.9.1

Source Codes:
```sh
conda create -n py38torch19 python=3.8
conda activate py38torch19
git clone --recursive https://github.com/DaShi-Git/masterThesis.git
cd masterThesis
pip install -r environment.txt

```
If `torch==1.9.1+cu111` could not be found, `torch==1.9.1` is here alternative.
## Installation
Source Codes:
```sh
python setup.py install

```
A new package called `matmul-cuda` will be installed to the conda environment.
## Appliication
