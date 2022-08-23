Master's Thesis: Automatic Code Generation for Kernel Fusion
===
In this project, an inference framework using Tensor Cores and Code Generation is developed to show how kernel can be fused under the circumstance of a fully connected network with flexible hidden channels and arbitrary activation function. The fused kernel is not precompiled but code generated according to the PyTorch model provided by the user. Some experiments are done to analyze how GEMM patterns, memory usage and data flowing can affect the performance. 
# Thesis Information
- Title:  `Automatic Code Generation for Kernel Fusion`
- Authors:  `Shi, Da`
- Supervisor: `Sebastian Weiss`
- Technical University of Munich
# Project Structure

# Environment
Required environment:
 - NVIDIA GPU with RTX, e.g. RTX20xx or RTX30xx (we use an RTX2080 Ti)
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
# Installation
Source Codes:
```sh
cd project
# go to the repository of masterThesis/project
python setup.py install

```
This step compiles the binding function between python and CUDA launch instruction, but the concrete kernel function is not compiled here, since the fused kernel is implemented in .cuh file, it will be compiled after the model structure is known.

A new package called `matmul-cuda` will be installed to the current conda environment.

After installation, user can call functions in the new package `matmul-cuda` by importing this package in a python file.
# Application
Running the function `matmul_cuda.evaluate_flexible_MLP(*params)` and providing the corresponding parameters can infer the provided model and input batches. An user interface is designed to enable the model trained with PyTorch framework to make faster inference on this project.

## Train a PyTorch fully-connected network

##进度

20.05 

从~/projects/masterThesis 上传一版，实现了matmul，MNK系数如果太大，比如大于6？就会提醒共享内存不足uses too much shared data (0x30000 bytes, 0xc000 max)，要注意在cuh内定义abcd，在共享内存。printf要特别注意数据类型，否则显示错误。
~/projects/tmp/masterThesis/extensionMatmul这一版没有上传，有a_frag[][]， 输出结果和python运算不一样，需要再看一下printf是否数据格式正确。是可运行状态

25.05

从~/projects/tmp2/masterThesis 上传一版，实现了flexible MLP， static MLP。目前都在一个block里面计算，需要看看如何用多个block， Seba说block之间不需要通信。放弃a_frag[]形式，因为需要用static define，还需 验证。__share memory定义时也需要static，所以只在kernel最开始定义了，后面复用，适用最大情况。

23.08
从~/projects/tmp2/masterThesis 上传一版，实现了灵活fc和多block推理。目前遇到的问题有load d和输出output时最后循环有一半的threads没有工作。用shuffle时总是有127个数字错误
