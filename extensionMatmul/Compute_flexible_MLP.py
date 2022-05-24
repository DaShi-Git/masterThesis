import torch
import sys
import os
sys.path.insert(0, os.getcwd())
#import lltm
#from matmul import Matmul #both import work from importing lltm nn model from lltm.py, see lines below
import time
import matmul_cuda
from utils.arbitaryActivation import writeActivation

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

hiddenStructure = [2, 1, 2, 1]

a = torch.ones(32, 32, device=cuda_device)
a01 = torch.ones(32, 32, device=cuda_device)*0.1

b01 = torch.ones(16*hiddenStructure[1], 16*2, device=cuda_device)*0.1
b02 = torch.ones(16*hiddenStructure[2], 16*hiddenStructure[1], device=cuda_device)*0.1
b03 = torch.ones(16*hiddenStructure[3], 16*hiddenStructure[2], device=cuda_device)*0.1
b05 = torch.ones(32, 32, device=cuda_device)*0.1
b05 = torch.ones(32, 32, device=cuda_device)*0.1

# print((((a.mm(a)+0).mm(b05)+0).mm(b05)+0).mm(b05).mm(b05))
#print((((a.mm(a)+0).mm(b01)+0).mm(b02)+0))
out = b03@b02@b01@a01@a
print(out)
print(out.size())
print(out.dtype)
