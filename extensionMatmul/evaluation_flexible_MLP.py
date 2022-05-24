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
"return z>0.0? z*z:0.0;",
"}"]
writeActivation(activation, truncate=True)
batch_size = 16
input_features = 32
state_size = 128
HIDDEN_CHANNELS = 16
# Note the device=cuda_device arguments here
# X = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)
# X1 = torch.reshape(X, (HIDDEN_CHANNELS, HIDDEN_CHANNELS))
# X1 = X1.transpose(0,1)
# X1 = torch.reshape(X, (HIDDEN_CHANNELS*HIDDEN_CHANNELS,1))

X = torch.randn(HIDDEN_CHANNELS, HIDDEN_CHANNELS, device=cuda_device)*1
X = torch.round(X)
h = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)
C = torch.randn(4,4, device=cuda_device)
matmul_cuda.cleanup
# output = matmul_cuda.evaluate_static_MLP(X, h, C)
output = matmul_cuda.evaluate_flexible_MLP(X, h, C, torch.Tensor([2.0,3.0,4.0]), 2)
#print(output)
#X = X.half()
#print("hii", X)
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
for i in range(HIDDEN_CHANNELS*HIDDEN_CHANNELS):
    h[i][0] = 1.0
#print(X.mul(h))
X1 = torch.reshape(X, (HIDDEN_CHANNELS, HIDDEN_CHANNELS)).half()
h = torch.reshape(h, (HIDDEN_CHANNELS, HIDDEN_CHANNELS)).half()
# print("/n")
# print(X.transpose(0,1).mul(h))
