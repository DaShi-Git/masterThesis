import torch
import sys
import os
sys.path.insert(0, os.getcwd())
#import lltm
#from matmul import Matmul #both import work from importing lltm nn model from lltm.py, see lines below
import time
import matmul_cuda
from utils.arbitaryActivation import writeActivation
from utils.arbitaryHiddenChannels import writeHiddenChannels

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU
activation = ["template <typename scalar_t>",
"__device__ __forceinline__ scalar_t arbiacti(scalar_t z) {",
"return z>0.0? z*z:0.0;",
"}"]
activation2 = ["template <typename scalar_t>",
"__device__ __forceinline__ scalar_t arbiacti2(scalar_t z) {",
"return sinf(z);",
"}"]
hiddenChannels = ["const int hiddenChannels[3] = {2, 1, 3};"]
writeActivation(activation, truncate=True)
writeHiddenChannels(hiddenChannels, truncate=True)
batch_size = 16
input_features = 32
state_size = 128
HIDDEN_CHANNELS = 32
# Note the device=cuda_device arguments here
# X = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)
# X1 = torch.reshape(X, (HIDDEN_CHANNELS, HIDDEN_CHANNELS))
# X1 = X1.transpose(0,1)
# X1 = torch.reshape(X, (HIDDEN_CHANNELS*HIDDEN_CHANNELS,1))

X1 = torch.randn(32*32,1, device=cuda_device)*1
# for i in range(32*32):
#     X1[i][0] = i*0.01
X2 = torch.ones(16*32,1, device=cuda_device)*0.1
X3 = torch.randn(48*16,1, device=cuda_device)*1
# for i in range(32*16):
#     X2[i][0] = i*0.01
X = torch.cat((X1, X2, X3), 0)
h = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)*1
# for i in range(32*32):
#     h[i][0] = i*0.01
C = torch.randn(4,4, device=cuda_device)
matmul_cuda.cleanup
# output = matmul_cuda.evaluate_static_MLP(X, h, C)
output = matmul_cuda.evaluate_flexible_MLP(X, h, C, torch.Tensor([2]), 2, 2, 48, 32, activation2)
#print(h)
#X = X.half()
#print("hii", X)


X1 = torch.reshape(X1, (HIDDEN_CHANNELS, HIDDEN_CHANNELS)).half()
X2 =torch.reshape(X2, (16, 32)).half()
X3 =torch.reshape(X3, (48, 16)).half()
h = torch.reshape(h, (HIDDEN_CHANNELS, HIDDEN_CHANNELS)).half()
# print("/n")
#print(h[0])
output1 = X1.mm(h.transpose(0, 1))

#print(output1.transpose(0, 1))
 
output2 = X2.mm(output1)
output3 = X3.mm(output2).float()
s = 0
#for i in range(16):
for i in range(48):
    for j in range(32):
        if (abs((output3-output)[i][j]) > 0.1):
            s=s+1
        #print((output2-output).ceil()[i][j], i*32+j, output2[i][j], output[i][j])
        print(i*32+j, (output3-output)[i][j], output3[i][j], output[i][j])
print("false number", s)
