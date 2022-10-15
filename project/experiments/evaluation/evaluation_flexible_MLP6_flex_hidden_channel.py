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
cuda_device = torch.device("cuda:0")  # device object representing GPU

def get_model_param(model_structure):
    with torch.no_grad():
        model = model_structure()
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
        timestamp1 = time.time()
        model.half()
        
        
        tmpoutput = model(h)
        timestamp2 = time.time()
        outputgroundModel = tmpoutput.float().transpose(0, 1).cpu()
        print("pytorch time half: ", (timestamp2-timestamp1)*1000)
   
        Xin = torch.empty_like(Xmodel, device=cuda_device, requires_grad=False)
        for i in range(Xmodel.size(1)):
            Xin[0][i] = Xmodel[0][i]

        Biasin = torch.empty_like(Biasmodel, device=cuda_device, requires_grad=False)
        for i in range(Biasmodel.size(1)):
            Biasin[0][i] = Biasmodel[0][i]
    return Xin, Biasin, outputgroundModel

with open('activationLibrary.json', 'r') as json_file:
    activation1 = json.load(json_file)["ReLU"]
    #activation2 = json.load(json_file)["LeaklyReLU"]
activation1 = activation1+["__device__ half (*activation[4])(half) = {ReLU, ReLU, ReLU, ReLU};"]
hiddenlayer=6
hiddstring = "const int hiddenChannels["+str(hiddenlayer+1)+"] = {8"
for i in range(hiddenlayer):
    hiddstring = hiddstring+",8"
hiddstring = hiddstring+"};"
#hiddstring = "const int hiddenChannels[5] = {8, 8, 8, 8, 8};"
hiddenChannels = [hiddstring]
#hiddenChannels = ["const int hiddenChannels[4] = {2, 8, 4, 2};","", "__device__ half (*activation[3])(half) = {ReLU, ReLU, ReLU};"]
Cin = 8 #first layer
Cout = 8 #last layer

# define the total input batch size
batchsizeTotal=20#13056*100
#batch_size = 32
blockdim = 384
griddim = 8


hInput = torch.randn(Cin*16,batchsizeTotal, device=cuda_device)*1 # input matrix to kernel
h = copy.deepcopy(hInput).transpose(0, 1).half() ## input matrix to model
padInput = nn.ZeroPad2d((0, (16-hInput.size(1)%16)%16, 0, 0))  #padding left, right, top, down

hInput = padInput(hInput)
batchsizeAfterPadding = hInput.size(1)
print("Input size", hInput.size(), h.size())
hInput = torch.reshape(hInput.transpose(0, 1), (1, Cin*16*hInput.size(1)))

# get model parameters
Xin, Biasin, outputgroundModel = get_model_param(ModelClass)


matmul_cuda.cleanup
Ctest  = torch.ones(2, 2, device=cuda_device)#.half()
###output = matmul_cuda.evaluate_flexible_MLP(Cn, Xin, hin, C16, torch.Tensor([4, 1, 4]), 2, 2, Cout*16, 32, activation2)
timestamp3  =  time.time()
for i in range(1):
    output = matmul_cuda.evaluate_flexible_MLP(Ctest, Xin, hInput, Biasin, hiddenChannels, batchsizeAfterPadding, 2, 1, Cout*16*batchsizeAfterPadding, activation1, blockdim, griddim)

timestamp4  =  time.time()
print("kernel time half: ", (timestamp4-timestamp3)*1000)

output = torch.reshape(output, (batchsizeAfterPadding, Cout*16)).transpose(0, 1)[:, :batchsizeTotal]

output = output.cpu()


# correctness check
outputground = outputgroundModel
# s = 0
# for i in range(outputground.size(0)):

#     for j in range(outputground.size(1)):
#         #if (abs(np.divide((outputground-output)[i][j].numpy(), outputground[i][j].numpy())) > 0.2):
#         if (abs((outputground-output)[i][j]) > 0.001):
#             s=s+1
#             #print(i*32+j,"\t diff:", (outputground-output)[i][j],"\t","\t", outputground[i][j], output[i][j])
#         #print(i*32+j, (X-output)[i][j], h[i][j], output[i][j])
#         ######print((output3-output).ceil()[i][j], i*32+j, output3[i][j], output[i][j])
# print("false number", s)
# print("kernel output size", output.size())
# print("pytorch output size", outputground.size())

