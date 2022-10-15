import torch
import torch.nn as nn
import sys
import os
import numpy as np
import copy
sys.path.insert(0, os.getcwd())
import time
import matmul_cuda
from utils.arbitaryActivation import writeActivation
from utils.arbitaryHiddenChannels import writeHiddenChannels
from designModel.train_model import ModelClass
import json
assert torch.cuda.is_available()
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu")  # device object representing GPU




batchsizeTotal=180000*10#13056*100
Cin = 4


h = torch.randn(batchsizeTotal,Cin*16, device=cpu_device)*1 # input matrix to kernel

model = ModelClass()
model.eval()

model.to(cuda_device)
h = h.to(cuda_device)
timestamp1 = time.time()


tmpoutput = model(h)
timestamp2 = time.time()
#outputgroundModel = tmpoutput.float().transpose(0, 1).cpu()
print("pytorch 32 time half: ", (timestamp2-timestamp1)*1000)
timestamp1 = time.time()
model.half()


tmpoutput = model(h.half())
timestamp2 = time.time()
#outputgroundModel = tmpoutput.float().transpose(0, 1).cpu()
print("pytorch 16 time half: ", (timestamp2-timestamp1)*1000)


