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


with open('activationLibrary.json', 'r') as json_file:
    activation1 = json.load(json_file)["ReLU"]
    #activation2 = json.load(json_file)["LeaklyReLU"]
activation1 = activation1+["__device__ half (*activation[4])(half) = {ReLU, ReLU, ReLU, ReLU};"]
hiddenlayer=4*10
hiddstring = "const int hiddenChannels["+str(hiddenlayer+1)+"] = {1"
for i in range(hiddenlayer):
    hiddstring = hiddstring+",1"
hiddstring = hiddstring+"};"
#hiddstring = "const int hiddenChannels[5] = {8, 8, 8, 8, 8};"
hiddenChannels = [hiddstring]
#hiddenChannels = ["const int hiddenChannels[4] = {2, 8, 4, 2};","", "__device__ half (*activation[3])(half) = {ReLU, ReLU, ReLU};"]
Cin = 1 #first layer
Cout = 1 #last layer

batchsizeTotal=200000*10#13056*100
#batch_size = 32
hInput = torch.randn(Cin*16,batchsizeTotal, device=cuda_device) # input matrix to kernel

h = copy.deepcopy(hInput).transpose(0, 1) ## input matrix to model
def getKernelInput(Input, Cin):
    
    padInput = nn.ZeroPad2d((0, (16-Input.size(1)%16)%16, 0, 0))  #padding left, right, top, down
    Input = padInput(Input)
    #print(Input.size())
    batchsizeAfterPadding = Input.size(1)
    h = torch.reshape(Input.transpose(0, 1), (1, Cin*16*Input.size(1)))
    return h, batchsizeAfterPadding


with torch.no_grad():
    model = ModelClass()
    for i, param_tensor in enumerate(model.state_dict()):
        if i ==0 or i == 1: #注意这里！！
            if model.state_dict()[param_tensor].dim() == 2:
                tmp_tensor = model.state_dict()[param_tensor]
                padWeights = nn.ZeroPad2d((0, (16-tmp_tensor.size(1)%16)%16, 0, (16-tmp_tensor.size(0)%16)%16))  #padding left, right, top, down
                tmp_tensor = padWeights(tmp_tensor)
                Xmodel = torch.reshape(tmp_tensor, (1, tmp_tensor.size(0)*tmp_tensor.size(1)))
            else:
                tmp_bias = model.state_dict()[param_tensor]
                padBias = nn.ConstantPad1d((0, (16-tmp_bias.size(0)%16)%16), 0.0)
                tmp_bias = padBias(tmp_bias)
                Biasmodel = torch.reshape(tmp_bias, (1, tmp_bias.size(0)))
                #print(tmp_bias.size())
        else:
            if model.state_dict()[param_tensor].dim() == 2:
                tmp_tensor = model.state_dict()[param_tensor]
                padWeights = nn.ZeroPad2d((0, (16-tmp_tensor.size(1)%16)%16, 0, (16-tmp_tensor.size(0)%16)%16))  #padding left, right, top, down
                tmp_tensor = padWeights(tmp_tensor)
                tmp_tensor = torch.reshape(tmp_tensor, (1, tmp_tensor.size(0)*tmp_tensor.size(1)))
                Xmodel = torch.cat((Xmodel, tmp_tensor), 1)
                
            else:
                tmp_bias = model.state_dict()[param_tensor]
                padBias = nn.ConstantPad1d((0, (16-tmp_bias.size(0)%16)%16), 0.0)
                tmp_bias = padBias(tmp_bias)
                tmp_bias = torch.reshape(tmp_bias, (1, tmp_bias.size(0)))
                Biasmodel = torch.cat((Biasmodel, tmp_bias), 1)
                #print(tmp_bias.size())

    model.eval()
    
    model.to(cuda_device)
    h = h.to(cuda_device)


    # timestamp1 = time.time()
    # for i in range(100):
    #     tmpoutput = model(h)
    # timestamp2 = time.time()
    # #outputgroundModel = tmpoutput.float().transpose(0, 1).cpu()
    # print("pytorch 32 time half: ", (timestamp2-timestamp1)*1000)

    h = h.half()
    model.half()
    timestamp1 = time.time()
    
    for i in range(100):
        tmpoutput = model(h)
    timestamp2 = time.time()
    #outputgroundModel = tmpoutput.float().transpose(0, 1).cpu()
    print("pytorch 16 time half: ", (timestamp2-timestamp1)*1)

matmul_cuda.cleanup


Xin = torch.empty_like(Xmodel, device=cuda_device, requires_grad=False)
for i in range(Xmodel.size(1)):
    Xin[0][i] = Xmodel[0][i]

Biasin = torch.empty_like(Biasmodel, device=cuda_device, requires_grad=False)
for i in range(Biasmodel.size(1)):
    Biasin[0][i] = Biasmodel[0][i]



Ctest  = torch.ones(2, 2, device=cuda_device)#.half()
###output = matmul_cuda.evaluate_flexible_MLP(Cn, Xin, hin, C16, torch.Tensor([4, 1, 4]), 2, 2, Cout*16, 32, activation2)
timestamp3  =  time.time()
for i in range(100):
    Input, batchsizeAfterPadding = getKernelInput(hInput, Cin)
    
    output = matmul_cuda.evaluate_flexible_MLP(Ctest, Xin, Input, Biasin, hiddenChannels, batchsizeAfterPadding, 2, 1, Cout*16*batchsizeAfterPadding, activation1)
    #print(output.size())
timestamp4  =  time.time()
print("kernel time half: ", (timestamp4-timestamp3)*1)

#

