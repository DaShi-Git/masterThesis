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


#hiddstring = "const int hiddenChannels[3] = {2, 3, 2};"
hiddenChannels = [hiddstring]
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
X = torch.randn(32*32,1, device=cuda_device)*1
X1 = torch.reshape(X, (HIDDEN_CHANNELS, HIDDEN_CHANNELS))
h = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)*1
hreshape = torch.reshape(h, (HIDDEN_CHANNELS, HIDDEN_CHANNELS))
outputground = X1.half().mm(hreshape.transpose(0, 1).half())
#print(outputground)
for i in range(hiddenlayer):
    Xtmp = torch.randn(32*32, 1, device=cuda_device)*0.2
    X = torch.cat((X, Xtmp), 0)

    X1tmp = torch.reshape(Xtmp, (HIDDEN_CHANNELS, HIDDEN_CHANNELS))
    outputground = X1tmp.half().mm(outputground)
    #outputground = torch.sin(outputground)


# X1 = torch.ones(32*32,1, device=cuda_device)*1
# # for i in range(32*32):
# #     X1[i][0] = i*0.01
# X2 = torch.ones(16*32,1, device=cuda_device)*0.1
# X3 = torch.ones(48*16,1, device=cuda_device)*1
# # for i in range(32*16):
# #     X2[i][0] = i*0.01
# X = torch.cat((X1, X2, X3), 0)
# h = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)*1
# for i in range(32*32):
#     h[i][0] = i*0.01
C = torch.randn(4,4, device=cuda_device)
test = torch.randn(4,4, device=cuda_device)
matmul_cuda.cleanup


Cout = 8
Cin = 8

Xn = torch.ones(Cout*16*Cin*16, 1, device=cuda_device)*1.0
hn = torch.ones(Cin*16*2*16, 1, device=cuda_device)*1.0


X16 = torch.ones(Cout*16*Cin*16, 1, device=cuda_device)*1.0
for i in range(Cout*16):
    for j in range(Cin*16):
        if i == j:
            X16[i*Cout*16+j][0] = 1.0
        else:
            X16[i*Cout*16+j][0] = 0.0

h16 = torch.ones(16*16, 1, device=cuda_device)*1.0
for i in range(16*16):
    h16[i][0] = (i+1)*1
h16 = torch.reshape(h16, (16, 16))
h16 = torch.cat( (h16,h16),0)
h16 = torch.cat( (h16,h16),1)

h16 = torch.cat( (h16,h16),0)
h16 = torch.cat( (h16,h16),0)
h16 = torch.reshape(h16.transpose(0, 1), (Cin*16*2*16, 1))
#h16 = torch.reshape(h16, (Cin*16*2*16, 1))

C16 = torch.ones(16*16, 1, device=cuda_device)*1.0
for i in range(16*16):
    C16[i][0] = (i+1)*0
C16 = torch.reshape(C16, (16, 16))
C16 = torch.cat( (C16,C16),0)
C16 = torch.cat( (C16,C16),1)

C16 = torch.cat( (C16,C16),0)
C16 = torch.cat( (C16,C16),0)
C16 = torch.reshape(C16.transpose(0, 1), (Cin*16*2*16, 1))
#C16 = torch.reshape(C16, (Cin*16*2*16, 1))


Xn1 = torch.reshape(Xn, (Cout*16, Cin*16))
#h = torch.randn(HIDDEN_CHANNELS*HIDDEN_CHANNELS,1, device=cuda_device)*1
hnreshape = torch.reshape(hn, (32, Cin*16))
outputgroundn = Xn1.half().mm(hnreshape.transpose(0, 1).half())
# output = matmul_cuda.evaluate_static_MLP(X, h, C)
Cn = torch.ones(1*16*1*16, 1, device=cuda_device)*2.0
for i in range(1*16*1*16):
    Cn[i][0] = i+1

# C16 = torch.ones(Cout*16*2*16, 1, device=cuda_device)*0.1
# h16 = torch.ones(Cin*16*2*16, 1, device=cuda_device)*1
output = matmul_cuda.evaluate_flexible_MLP(C, X16, h16, C16, torch.Tensor([2, 1, 2, 1]), 2, 2, 4*16, 32, activation2)
#print(h16)
#X = X.half()
#print("hii", X)


# X1 = torch.reshape(X1, (HIDDEN_CHANNELS, HIDDEN_CHANNELS))
# X2 =torch.reshape(X2, (16, 32))
# X3 =torch.reshape(X3, (48, 16))
# h = torch.reshape(h, (HIDDEN_CHANNELS, HIDDEN_CHANNELS))
# # print("/n")
# #print(h[0])
# output1 = X1.mm(h.transpose(0, 1))

# #print(output1.transpose(0, 1))
 
# output2 = X2.mm(output1)
# output3 = X3.mm(output2)

# output3 = outputground.float()
#output3 = outputgroundn.float()
s = 0
#for i in range(2*16):

    #for j in range(32):
        # if (abs((output3-output)[i][j]) > 0.1):
        #     s=s+1
        #print(i*32+j, (output3-output)[i][j], output3[i][j], output[i][j])
        #print((output3-output).ceil()[i][j], i*32+j, output3[i][j], output[i][j])
    #print("false number", s, torch.sin(torch.tensor(2.0)))

# for i in range(128):
#     for j in range(128):
#         if i==j:
#             print(i, X16[i*Cout*16+j][0])
