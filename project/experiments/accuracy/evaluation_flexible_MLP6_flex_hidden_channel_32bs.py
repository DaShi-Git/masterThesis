import torch
import torch.nn as nn
import sys
from PIL import Image
import os
import numpy as np
import copy
sys.path.insert(0, os.getcwd())
import time
import matmul_cuda
from utils.arbitaryActivation import writeActivation
from utils.arbitaryHiddenChannels import writeHiddenChannels
from designModel.train_model import ModelClass
from experiments.accuracy.train_model import ModelClassTest, ModelClassTest1
import json
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU


with open('activationLibrary.json', 'r') as json_file:
    activation1 = json.load(json_file)["ReLU"]
    #activation2 = json.load(json_file)["LeaklyReLU"]
activation1 = activation1+["__device__ half (*activation[4])(half) = {ReLU, ReLU, ReLU, ReLU};"]
# hiddenlayer=2
# hiddstring = "const int hiddenChannels["+str(hiddenlayer+1)+"] = {2"
# for i in range(hiddenlayer):
#     hiddstring = hiddstring+",2"
# hiddstring = hiddstring+"};"
hiddstring = "const int hiddenChannels[5] = {2, 4, 6, 8, 8};"
#hiddstring = "const int hiddenChannels[5] = {2*16, 8*16, 8*16, 6*16, 2*16};"
hiddenChannels = [hiddstring]
#hiddenChannels = ["const int hiddenChannels[4] = {2, 8, 4, 2};","", "__device__ half (*activation[3])(half) = {ReLU, ReLU, ReLU};"]
Cin = 2 #first layer
Cout = 8 #last layer

batchsizeTotal=192*4
batch_size = 32


hInput = torch.ones(Cin*16,batchsizeTotal, device=cuda_device)*0.5 # input matrix to kernel
h = copy.deepcopy(hInput).transpose(0, 1).half() ## input matrix to model
padInput = nn.ZeroPad2d((0, (16-hInput.size(1)%16)%16, 0, 0))  #padding left, right, top, down
batchsizeAfterPadding = hInput.size(1)
hInput = padInput(hInput)
print("Input size", hInput.size(), h.size())
hInput = torch.reshape(hInput.transpose(0, 1), (1, Cin*16*hInput.size(1)))


with torch.no_grad():
    #model = ModelClass()
    model  = ModelClassTest()
    #torch.save(model.state_dict(),'/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/net.pth.tar')
    
    model.load_state_dict(torch.load('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/net1_4layers.pth.tar'))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    for i, param_tensor in enumerate(model.state_dict()):
        if i<=7:
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
        elif i == 8:
            fc5_weight = model.state_dict()[param_tensor]
        elif i == 9:
            fc5_bias = model.state_dict()[param_tensor]
                
    #print(Xmodel)
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

Biasin = torch.empty_like(Biasmodel, device=cuda_device, requires_grad=False)
for i in range(Biasmodel.size(1)):
    Biasin[0][i] = Biasmodel[0][i]

Bias = torch.ones((1, 128*12), device=cuda_device)*0.0
#Bias = torch.ones((1, Biasmodel.size(0)), device=cuda_device)*0.0
# for i in range(Biasmodel.size(0)):
#     Bias[0][i] = 0 #Biasmodel[i]

Ctest  = torch.ones(2, 128, device=cuda_device)#.half()
###output = matmul_cuda.evaluate_flexible_MLP(Cn, Xin, hin, C16, torch.Tensor([4, 1, 4]), 2, 2, Cout*16, 32, activation2)

timestamp3  =  time.time()
output = matmul_cuda.evaluate_flexible_MLP(Ctest, Xin, hInput, Biasin, hiddenChannels, batchsizeAfterPadding, 2, 1, Cout*16*batchsizeAfterPadding, activation1)
timestamp4  =  time.time()
print("kernel time half: ", (timestamp4-timestamp3)*1000)
#

# for i in range(Cout*16):
#     i += 0
#     print(i, output[0][i+0*32], outputground[i][0])
output = torch.reshape(output, (batchsizeAfterPadding, Cout*16)).transpose(0, 1)[:, :batchsizeTotal]

output = output.cpu()
#outputground = outputground.cpu().float()
outputground = outputgroundModel
s = 0
pytorchout = []
kernelout = []
for i in range(outputground.size(0)):

    for j in range(1):
        #if (abs(np.divide((outputground-output)[i][j].numpy(), outputground[i][j].numpy())) > 0.2):
        if (abs((outputground-output)[i][j]) > 0.001):
            s=s+1
        print(i,"\t diff:", (outputground-output)[i][j],"\t","\t", outputground[i][j], output[i][j])
        pytorchout.append(outputground[i][j].item())
        kernelout.append(output[i][j].item())
        #print(i*32+j, (X-output)[i][j], h[i][j], output[i][j])
        ######print((output3-output).ceil()[i][j], i*32+j, output3[i][j], output[i][j])
print("false number", s)
print("kernel output size", output.size())
print("pytorch output size", outputground.size())
# for i in range(128):
#     for j in range(128):
#         if i==j:
#             print(i, X16[i*Cout*16+j][0])

pytorchout = torch.Tensor(pytorchout)
kernelout = torch.Tensor(kernelout)
loss_func = nn.MSELoss()
loss = loss_func(pytorchout,kernelout)
print('loss between PyTorch and fused kernel',loss)
model = ModelClassTest()
model1=ModelClassTest1()

model.load_state_dict(torch.load('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/net1_4layers.pth.tar'))
model1.load_state_dict(torch.load('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/net1.pth.tar'))
for i, param_tensor in enumerate(model1.state_dict()):
    # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #print(param_tensor, "\t")
    if i<=7:
        kk = 0
    elif i == 8:
        fc5_weight = model1.state_dict()[param_tensor]
    elif i == 9:
        fc5_bias = model1.state_dict()[param_tensor]
# model.eval().to(cuda_device)
# model.half()
# output = model(torch.ones(32).to(cuda_device).half()*0.5)
# output=torch.reshape(output,  (8*16, 1)).float().cpu()
output = kernelout
output=torch.reshape(output,  (8*16, 1))
print(output)
output = fc5_weight.mm(output)#+fc5_bias
output = torch.reshape(output, (4333, 3250))
output  = output.numpy().astype(np.uint8)
print(output)


I = Image.open('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/albert.jpg')
I = np.array(I)
print(I)
img = Image.fromarray((output))
img.save('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/result1_FP16_kernel.jpg')