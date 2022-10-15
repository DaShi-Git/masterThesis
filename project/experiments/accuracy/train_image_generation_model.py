import numpy as np
from collections import OrderedDict
import os
import sys
import torch
import time
from PIL import Image
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
cuda_device = torch.device("cuda")
I = Image.open('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/albert.jpg')
I = np.array(I).astype(np.float32)
#print(I)

#img = Image.fromarray((I))
#img.save('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/result.jpg')

class ModelClassTest(nn.Module):
    def __init__(self):
        super(ModelClassTest, self).__init__()
        #input dim, output dim
        # self.fc1 = nn.Linear(2*16, 8*16-1, bias=False)
        # self.fc2 = nn.Linear(8*16-1, 8*16-1, bias=False)
        # self.fc3 = nn.Linear(8*16-1, 2*16, bias=False)

        # self.fc1 = nn.Linear(2*16, 4*16, bias=False)
        # self.fc2 = nn.Linear(4*16, 6*16, bias=False)
        # self.fc3 = nn.Linear(6*16, 8*16, bias=False)
        # self.fc4 = nn.Linear(8*16, 8*16, bias=False)

        # self.fc5 = nn.Linear(8*16, 4333*3250, bias=False)
        self.fc1 = nn.Linear(2*16, 4*16-1, bias=True)
        self.fc2 = nn.Linear(4*16-1, 6*16-1, bias=True)
        self.fc3 = nn.Linear(6*16-1, 8*16, bias=True)
        self.fc4 = nn.Linear(8*16, 8*16, bias=True)
        #self.fc4 = nn.Linear(8*16, 8*16, bias=True)

        #self.fc5 = nn.Linear(8*16, 4333*3250, bias=True)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        #x = self.fc5(x)
       

        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)

        #x = self.fc4(x)
        return x

class ModelClassTest1(nn.Module):
    def __init__(self):
        super(ModelClassTest1, self).__init__()
        #input dim, output dim
        # self.fc1 = nn.Linear(2*16, 8*16-1, bias=False)
        # self.fc2 = nn.Linear(8*16-1, 8*16-1, bias=False)
        # self.fc3 = nn.Linear(8*16-1, 2*16, bias=False)

        # self.fc1 = nn.Linear(2*16, 4*16, bias=False)
        # self.fc2 = nn.Linear(4*16, 6*16, bias=False)
        # self.fc3 = nn.Linear(6*16, 8*16, bias=False)
        # self.fc4 = nn.Linear(8*16, 8*16, bias=False)

        # self.fc5 = nn.Linear(8*16, 4333*3250, bias=False)
        self.fc1 = nn.Linear(2*16, 4*16-1, bias=True)
        self.fc2 = nn.Linear(4*16-1, 6*16-1, bias=True)
        self.fc3 = nn.Linear(6*16-1, 8*16, bias=True)
        self.fc4 = nn.Linear(8*16, 8*16, bias=True)
        #self.fc4 = nn.Linear(8*16, 8*16, bias=True)

        self.fc5 = nn.Linear(8*16, 4333*3250, bias=False)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.fc5(x)
       

        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)

        #x = self.fc4(x)
        return x

# model = ModelClassTest1()
# optimizer = torch.optim.SGD(model.parameters(), lr=2)
# loss_func = nn.MSELoss()
# I = torch.Tensor(I).reshape(-1)
# #I = torch.ones(1,8*16)*5
# losses  = []
# counter = []
# for iteration in range(2000):
#     optimizer.zero_grad()
#     input = torch.ones(32)*0.5
#     output = model(input)
#     #output = torch.reshape(output, (4333, 3250))

    
#     loss = loss_func(output, I)
#     loss.backward()

#     optimizer.step()

#     print(loss.item())
#     losses.append(loss.item())
#     counter.append(iteration)
#     if loss.item()  < 0.01:
#         break

# torch.save(model.state_dict(),'/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/net1.pth.tar')
# np.savetxt('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/save_losses.txt', np.array(losses), fmt='%f')
# np.savetxt('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/save_counter.txt', np.array(counter), fmt='%d')

# lossestxt = np.loadtxt('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/save_losses.txt', dtype=float)
# countertxt = np.loadtxt('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/save_counter.txt', dtype=int)

# print(lossestxt, countertxt)

# plt.scatter(counter, losses, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
# plt.ylabel('MSE Loss',fontsize=12,color='b')
# plt.xlabel('Iterations',fontsize=12,color='b')
# plt.title('Training Process', fontsize=20)
# plt.subplots_adjust(left=0.12, bottom=0.12, right=0.15, top=0.15, wspace=None, hspace=None)
# plt.savefig('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/plot1.jpg')

# #single-precision
# model = ModelClassTest1()
# model.load_state_dict(torch.load('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/net1.pth.tar'))
# model.eval()
# output = model(torch.ones(32)*0.5)
# output=torch.reshape(output,  (4333, 3250)).detach().numpy().astype(np.uint8)
# print(output)

# I = Image.open('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/albert.jpg')
# I = np.array(I)
# print(I)
# img = Image.fromarray((output))
# img.save('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/result1_FP32.jpg')


# #half precision:
# def copyStateDict(state_dict):
#     if list(state_dict.keys())[0].startswith('module'):
#         start_idx = 1
#     else:
#         start_idx = 0
#     new_state_dict = OrderedDict()
#     for k,v in state_dict.items():
#         name = '.'.join(k.split('.')[start_idx:])

#         new_state_dict[name] = v
#     return new_state_dict

# state_dict = torch.load('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/net1.pth.tar')
# new_dict = copyStateDict(state_dict)

# keys=[]
# for k,v in new_dict.items():
#     if k.startswith('fc5'):    #将‘arc’开头的key过滤掉，这里是要去除的层的key
#         continue
#     keys.append(k)

# new_dict = {k:new_dict[k] for k in keys}
# print(new_dict)
# model = ModelClassTest()
# model.load_state_dict(new_dict)
# torch.save(model.state_dict(),'/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/net1_4layers.pth.tar')




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
model.eval().to(cuda_device)
model.half()
output = model(torch.ones(32).to(cuda_device).half()*0.5)
output=torch.reshape(output,  (8*16, 1)).float().cpu()
print(output)
#print(fc5_bias)
output = fc5_weight.mm(output)#+fc5_bias
output = torch.reshape(output, (4333, 3250))
output  = output.detach().numpy().astype(np.uint8)



# I = Image.open('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/albert.jpg')
# I = np.array(I)
# print(I)
# img = Image.fromarray((output))
# img.save('/home/dashi/projects/tmp5/masterThesis/project/experiments/accuracy/result1_FP16.jpg')


