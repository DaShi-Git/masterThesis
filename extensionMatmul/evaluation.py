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
activation = ["template <typename scalar_t>",
"__device__ __forceinline__ scalar_t arbiacti(scalar_t z) {",
"return z>0.0? z:8.0;//1.0 / (1.0 + exp(-z));",
"}"]
#arbitActivation = ArbitActivation()
writeActivation(activation, truncate=True)
batch_size = 16
input_features = 32
state_size = 128

# Note the device=cuda_device arguments here
X = torch.randn(5,4, device=cuda_device)
#X = torch.Tensor([3,4])
h = torch.randn(4,4, device=cuda_device)
C = torch.randn(4,4, device=cuda_device)
matmul_cuda.cleanup
output = matmul_cuda.evaluate(X, h, C)
#print(output)
print("hii")
#rnn = lltm.LLTM(input_features, state_size).to(cuda_device)
# rnn = Matmul(4,4).to(cuda_device)

# forward = 0
# backward = 0
# for i in range(2):
#     start = time.time()
#     output = rnn(X, h, C)
#     torch.cuda.synchronize()
#     forward += time.time() - start

#     start = time.time()
#     # (new_h.sum() + new_C.sum()).backward()
#     torch.cuda.synchronize()
#     backward += time.time() - start
#     print(i)
#     print(X)
#     print(output)

# print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))