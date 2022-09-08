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
cuda_device = torch.device("cuda")  # device object representing GPU


with open('activationLibrary.json', 'r') as json_file:
    activation1 = json.load(json_file)["ReLU"]
    #activation2 = json.load(json_file)["LeaklyReLU"]
activation1 = activation1+["__device__ half (*activation[3])(half) = {ReLU, ReLU, ReLU};"]
# hiddenlayer=2
# hiddstring = "const int hiddenChannels["+str(hiddenlayer+1)+"] = {2"
# for i in range(hiddenlayer):
#     hiddstring = hiddstring+",2"
# hiddstring = hiddstring+"};"
hiddstring = "const int hiddenChannels[4] = {2, 8, 4, 2};"
hiddenChannels = [hiddstring]
#hiddenChannels = ["const int hiddenChannels[4] = {2, 8, 4, 2};","", "__device__ half (*activation[3])(half) = {ReLU, ReLU, ReLU};"]
Cin = 2 #first layer
Cout = 2 #last layer

batchsizeTotal=192*5
batch_size = 32


hInput = torch.randn(Cin*16,batchsizeTotal, device=cuda_device)*1 # input matrix to kernel
h = copy.deepcopy(hInput).transpose(0, 1).half() ## input matrix to model
padInput = nn.ZeroPad2d((0, (16-hInput.size(1)%16)%16, 0, 0))  #padding left, right, top, down
batchsizeAfterPadding = hInput.size(1)
hInput = padInput(hInput)
print("Input size", hInput.size(), h.size())
hInput = torch.reshape(hInput.transpose(0, 1), (1, Cin*16*hInput.size(1)))


with torch.no_grad():
    model = ModelClass()
    for i, param_tensor in enumerate(model.state_dict()):
        if i ==0:
            if model.state_dict()[param_tensor].dim() == 2:
                tmp_tensor = model.state_dict()[param_tensor]
                padWeights = nn.ZeroPad2d((0, (16-tmp_tensor.size(1)%16)%16, 0, (16-tmp_tensor.size(0)%16)%16))  #padding left, right, top, down
                tmp_tensor = padWeights(tmp_tensor)
                Xmodel = torch.reshape(tmp_tensor, (1, tmp_tensor.size(0)*tmp_tensor.size(1)))
            else:
                Biasmodel = model.state_dict()[param_tensor]
        else:
            if model.state_dict()[param_tensor].dim() == 2:
                tmp_tensor = model.state_dict()[param_tensor]
                padWeights = nn.ZeroPad2d((0, (16-tmp_tensor.size(1)%16)%16, 0, (16-tmp_tensor.size(0)%16)%16))  #padding left, right, top, down
                tmp_tensor = padWeights(tmp_tensor)
                tmp_tensor = torch.reshape(tmp_tensor, (1, tmp_tensor.size(0)*tmp_tensor.size(1)))
                Xmodel = torch.cat((Xmodel, tmp_tensor), 1)
            else:
                tmp_bias = model.state_dict()[param_tensor]
                Biasmodel = torch.cat((Biasmodel, tmp_bias), 0)
        #print(Xmodel.size())
        #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        #print(param_tensor, "\t", model.state_dict()[param_tensor].dtype)


    timestamp1 = time.time()
    model.half()
    model.eval()
    
    model.to(cuda_device)
    
    tmpoutput = model(h)
    timestamp2 = time.time()
    outputgroundModel = tmpoutput.float().transpose(0, 1).cpu()
    print("pytorch time half: ", (timestamp2-timestamp1)*1000)

    # model.float()
    # model.eval()
    # timestamp1 = time.time()
    # #model.to(cuda_device)
    # h = h.float()
    # tmpoutput = model(h)
    # timestamp2 = time.time()
    # #outputgroundModel = tmpoutput.float().transpose(0, 1).cpu()
    # print("pytorch time float: ", timestamp2-timestamp1)
   

matmul_cuda.cleanup


Xin = torch.empty_like(Xmodel, device=cuda_device, requires_grad=False)
for i in range(Xmodel.size(1)):
    Xin[0][i] = Xmodel[0][i]

Bias = torch.ones((1, 128*2), device=cuda_device)*0.0
#Bias = torch.ones((1, Biasmodel.size(0)), device=cuda_device)*0.0
# for i in range(Biasmodel.size(0)):
#     Bias[0][i] = 0 #Biasmodel[i]

Ctest  = torch.ones(2, 2, device=cuda_device)#.half()
###output = matmul_cuda.evaluate_flexible_MLP(Cn, Xin, hin, C16, torch.Tensor([4, 1, 4]), 2, 2, Cout*16, 32, activation2)
output = matmul_cuda.evaluate_flexible_MLP(Ctest, Xin, hInput, Bias, hiddenChannels, batchsizeAfterPadding, 2, 1, Cout*16*batchsizeAfterPadding, activation1)
#

# for i in range(Cout*16):
#     i += 0
#     print(i, output[0][i+0*32], outputground[i][0])
output = torch.reshape(output, (batchsizeAfterPadding, Cout*16)).transpose(0, 1)[:, :batchsizeTotal]

output = output.cpu()
#outputground = outputground.cpu().float()
outputground = outputgroundModel
s = 0
for i in range(outputground.size(0)):

    for j in range(outputground.size(1)):
        #if (abs(np.divide((outputground-output)[i][j].numpy(), outputground[i][j].numpy())) > 0.2):
        if (abs((outputground-output)[i][j]) > 0.001):
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
