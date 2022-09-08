import torch
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

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU
activation = ["template <typename scalar_t>",
"__device__ __forceinline__ scalar_t arbiacti1(scalar_t z) {",
"return z>0.0? z:0.0;",
"}"]
activation1 = ["template <typename scalar_t>",
"__device__ __forceinline__ scalar_t arbiacti1(scalar_t z) {",
"return z>(half)0.0? z:(half)0.0;",
"}"]
activation2 = ["template <typename scalar_t>",
"__device__ __forceinline__ scalar_t arbiacti2(scalar_t z) {",
"return sinf(z);",
"}"]
# hiddenlayer=2
# hiddstring = "const int hiddenChannels["+str(hiddenlayer+1)+"] = {2"
# for i in range(hiddenlayer):
#     hiddstring = hiddstring+",2"
# hiddstring = hiddstring+"};"
hiddstring = "const int hiddenChannels[4] = {2, 8, 4, 2};"
hiddenChannels = [hiddstring]
Cin = 2 #first layer
Cout = 2 #last layer

batchsizeTotal=96*1
batch_size = 32


hInput = torch.randn(Cin*16,batchsizeTotal, device=cuda_device)*1
h = copy.deepcopy(hInput).transpose(0, 1).half()
hInput = torch.reshape(hInput.transpose(0, 1), (1, Cin*16*batchsizeTotal))


with torch.no_grad():
    model = ModelClass()
    for i, param_tensor in enumerate(model.state_dict()):
        if i ==0:
            if model.state_dict()[param_tensor].dim() == 2:
                tmp_tensor = model.state_dict()[param_tensor]
                Xmodel = torch.reshape(tmp_tensor, (1, tmp_tensor.size(0)*tmp_tensor.size(1)))
            else:
                Biasmodel = model.state_dict()[param_tensor]
        else:
            if model.state_dict()[param_tensor].dim() == 2:
                tmp_tensor = model.state_dict()[param_tensor]
                tmp_tensor = torch.reshape(tmp_tensor, (1, tmp_tensor.size(0)*tmp_tensor.size(1)))
                Xmodel = torch.cat((Xmodel, tmp_tensor), 1)
            else:
                tmp_bias = model.state_dict()[param_tensor]
                Biasmodel = torch.cat((Biasmodel, tmp_bias), 0)
        #print(Xmodel.size())
        #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        #print(param_tensor, "\t", model.state_dict()[param_tensor].dtype)


    
    model.half()
    model.eval()
    model.to(cuda_device)
    outputgroundModel = model(h).float().transpose(0, 1).cpu()

matmul_cuda.cleanup


Xin = torch.empty_like(Xmodel, device=cuda_device, requires_grad=False)
for i in range(Xmodel.size(1)):
    Xin[0][i] = Xmodel[0][i]

Bias = torch.ones((1, 128*2), device=cuda_device)*0.0
#Bias = torch.ones((1, Biasmodel.size(0)), device=cuda_device)*0.0
# for i in range(Biasmodel.size(0)):
#     Bias[0][i] = 0 #Biasmodel[i]

Ctest  = torch.ones(2, 2, dtype=torch.float)
###output = matmul_cuda.evaluate_flexible_MLP(Cn, Xin, hin, C16, torch.Tensor([4, 1, 4]), 2, 2, Cout*16, 32, activation2)
output = matmul_cuda.evaluate_flexible_MLP(Ctest, Xin, hInput, Bias, hiddenChannels, batchsizeTotal, 2, 1, Cout*16*batchsizeTotal, activation1)
#

# for i in range(Cout*16):
#     i += 0
#     print(i, output[0][i+0*32], outputground[i][0])
output = torch.reshape(output, (batchsizeTotal, Cout*16)).transpose(0, 1)
output = output.cpu()
#outputground = outputground.cpu().float()
outputground = outputgroundModel
s = 0
for i in range(outputground.size(0)):

    for j in range(outputground.size(1)):
        #if (abs(np.divide((outputground-output)[i][j].numpy(), outputground[i][j].numpy())) > 0.2):
        if (abs((outputground-output)[i][j]) > 0.2):
            s=s+1
        print(i*32+j,"\t diff:", (outputground-output)[i][j],"\t","\t", outputground[i][j], output[i][j])
        #print(i*32+j, (X-output)[i][j], h[i][j], output[i][j])
        ######print((output3-output).ceil()[i][j], i*32+j, output3[i][j], output[i][j])
print("false number", s)
print("kernel output size", output.size())
print("pytorch output size", outputground.size())
# for i in range(128):
#     for j in range(128):
#         if i==j:
#             print(i, X16[i*Cout*16+j][0])
