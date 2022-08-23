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
"__device__ __forceinline__ scalar_t arbiacti1(scalar_t z) {",
"return z>0.0? z*z:7.0;",
"}"]
activation2 = ["template <typename scalar_t>",
"__device__ __forceinline__ scalar_t arbiacti2(scalar_t z) {",
"return sinf(z);",
"}"]
hiddenlayer=2
hiddstring = "const int hiddenChannels["+str(hiddenlayer+1)+"] = {2"
for i in range(hiddenlayer):
    hiddstring = hiddstring+",2"
hiddstring = hiddstring+"};"


hiddstring = "const int hiddenChannels[4] = {2, 6, 4, 2};"
hiddenChannels = [hiddstring]
writeActivation(activation, truncate=True)
writeHiddenChannels(hiddenChannels, truncate=True)

input_features = 32
state_size = 128
HIDDEN_CHANNELS = 32
# Note the device=cuda_device arguments here
# X = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)
# X1 = torch.reshape(X, (HIDDEN_CHANNELS, HIDDEN_CHANNELS))
# X1 = X1.transpose(0,1)
# X1 = torch.reshape(X, (HIDDEN_CHANNELS*HIDDEN_CHANNELS,1))
batchsizeTotal=64
batch_size = 32
Cin = 2
Cout = 6

X = torch.randn(Cout*16, Cin*16, device=cuda_device)*0.5
Xin = torch.reshape(X, (1, Cout*16* Cin*16))
h = torch.randn(Cin*16,batchsizeTotal, device=cuda_device)*1
hin = torch.reshape(h.transpose(0, 1), (1, Cin*16*batchsizeTotal))
outputground = X.half().mm(h.half())
#print(outputground)

Cin = 6
Cout = 4

X2 = torch.randn(Cout*16, Cin*16, device=cuda_device)*0.9
Xin2 = torch.reshape(X2, (1, Cout*16* Cin*16))
Xin =  torch.cat((Xin, Xin2), 1)

outputground = X2.half().mm(outputground.half())
#print(outputground)


Cin = 4
Cout = 2

X3 = torch.randn(Cout*16, Cin*16, device=cuda_device)*0.9
Xin3 = torch.reshape(X3, (1, Cout*16* Cin*16))
Xin =  torch.cat((Xin, Xin3), 1)

outputground = X3.half().mm(outputground.half())
#print(outputground)
matmul_cuda.cleanup




# Xn = torch.ones(Cout*16*Cin*16, 1, device=cuda_device)*1.0
# hn = torch.ones(Cin*16*2*16, 1, device=cuda_device)*1.0


# X16 = torch.ones(Cout*16*Cin*16, 1, device=cuda_device)*1.0
# for i in range(Cout*16):
#     for j in range(Cin*16):
#         if i == j:
#             X16[i*Cout*16+j][0] = 1.0
#         else:
#             X16[i*Cout*16+j][0] = 0.0

# h16 = torch.ones(16*16, 1, device=cuda_device)*1.0
# for i in range(16*16):
#     h16[i][0] = (i+1)*1
# h16 = torch.reshape(h16, (16, 16))
# h16 = torch.cat( (h16,h16),0)
# h16 = torch.cat( (h16,h16),1)

# h16 = torch.cat( (h16,h16),0)
# h16 = torch.cat( (h16,h16),0)
# h16 = torch.reshape(h16.transpose(0, 1), (Cin*16*2*16, 1))
# #h16 = torch.reshape(h16, (Cin*16*2*16, 1))

C16 = torch.ones(16*16, 1, device=cuda_device)*1.0
# for i in range(16*16):
#     C16[i][0] = (i+1)*0
# C16 = torch.reshape(C16, (16, 16))
# C16 = torch.cat( (C16,C16),0)
# C16 = torch.cat( (C16,C16),1)

# C16 = torch.cat( (C16,C16),0)
# C16 = torch.cat( (C16,C16),0)
# C16 = torch.reshape(C16.transpose(0, 1), (Cin*16*2*16, 1))
# #C16 = torch.reshape(C16, (Cin*16*2*16, 1))


# Xn1 = torch.reshape(Xn, (Cout*16, Cin*16))
# #h = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)*1
# hnreshape = torch.reshape(hn, (32, Cin*16))
# outputgroundn = Xn1.half().mm(hnreshape.transpose(0, 1).half())
# # output = matmul_cuda.evaluate_static_MLP(X, h, C)
# Cn = torch.ones(1*16*1*16, 1, device=cuda_device)*2.0
# for i in range(1*16*1*16):
#     Cn[i][0] = i+1
Cn  = torch.ones(2, 2, dtype=torch.float)
# C16 = torch.ones(Cout*16*2*16, 1, device=cuda_device)*0.1
# h16 = torch.ones(Cin*16*2*16, 1, device=cuda_device)*1
###output = matmul_cuda.evaluate_flexible_MLP(Cn, Xin, hin, C16, torch.Tensor([4, 1, 4]), 2, 2, Cout*16, 32, activation2)
output = matmul_cuda.evaluate_flexible_MLP(Cn, Xin, hin, C16, torch.Tensor([4, 1, 4]), batchsizeTotal, 2, 1, Cout*16*batchsizeTotal, activation2)
#

# for i in range(Cout*16):
#     i += 0
#     print(i, output[0][i+0*32], outputground[i][0])
output = torch.reshape(output, (batchsizeTotal, Cout*16)).transpose(0, 1)

s = 0
for i in range(outputground.size(0)):

    for j in range(60,outputground.size(1)):
        if (abs((outputground-output)[i][j]) > 1.0):
        #if (abs((X-output)[i][j]) > 0.01):
            s=s+1
            print(i*32+j, (outputground-output)[i][j], outputground[i][j], output[i][j])
        #print(i*32+j, (X-output)[i][j], h[i][j], output[i][j])
        ######print((output3-output).ceil()[i][j], i*32+j, output3[i][j], output[i][j])
print("false number", s, torch.sin(torch.tensor(2.0)))
print(output.size())
print(outputground.size())
# for i in range(128):
#     for j in range(128):
#         if i==j:
#             print(i, X16[i*Cout*16+j][0])
