Master's Thesis: Automatic Code Generation for Kernel Fusion
===
Abstract:xxx
## Thesis Information
- Title:  `Automatic Code Generation for Kernel Fusion`
- Authors:  `Shi, Da`
- Supervisor: 'Sebastian Weiss'
- Full-preprint: [paper position]()
- Video: [video position]()
## Environment
Required environment:
 - NVIDIA GPU with RTX, e.g. RTX20xx or RTX30xx (we use an RTX2080)
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
## Application

##进度

20.05 

从~/projects/masterThesis 上传一版，实现了matmul，MNK系数如果太大，比如大于6？就会提醒共享内存不足uses too much shared data (0x30000 bytes, 0xc000 max)，要注意在cuh内定义abcd，在共享内存。printf要特别注意数据类型，否则显示错误。
~/projects/tmp/masterThesis/extensionMatmul这一版没有上传，有a_frag[][]， 输出结果和python运算不一样，需要再看一下printf是否数据格式正确。是可运行状态

