Master's Thesis: Automatic Code Generation for Kernel Fusion
===
In this project, an inference framework using Tensor Cores and Code Generation is developed to show how kernel can be fused under the circumstance of a fully connected network with flexible hidden channels and arbitrary activation function. The fused kernel is not precompiled but code generated according to the PyTorch model provided by the user. Some experiments are done to analyze how GEMM patterns, memory usage and data flowing can affect the performance. 
# Thesis Information
- Title:  `Automatic Code Generation for Kernel Fusion`
- Authors:  `Shi, Da`
- Supervisor: `Weiss, Sebastian`
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

## Evaluate the model
Source Codes:
```sh
python evaluation_flexible_MLP5.py

```
It reports the kernel run time, correctness and the activition function designed by user.
