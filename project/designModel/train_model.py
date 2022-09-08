import torch
from torch import optim
import torch.nn as nn
import numpy as numpy
from torch.nn import functional as F
import torch.fx

class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        #input dim, output dim
        # self.fc1 = nn.Linear(2*16, 8*16-1, bias=False)
        # self.fc2 = nn.Linear(8*16-1, 8*16-1, bias=False)
        # self.fc3 = nn.Linear(8*16-1, 2*16, bias=False)

        self.fc1 = nn.Linear(2*16, 8*16-1, bias=False)
        self.fc2 = nn.Linear(8*16-1, 8*16-1, bias=False)
        self.fc3 = nn.Linear(8*16-1, 6*16-1, bias=False)
        self.fc4 = nn.Linear(6*16-1, 2*16, bias=False)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)

        #x = self.fc4(x)
        return x



# class ModelClass(nn.Module):
#     def __init__(self):
#         super(ModelClass, self).__init__()
#         #input dim, output dim
#         self.fc1 = nn.Linear(2*16, 8*16-10, bias=False)
#         self.fc2 = nn.Linear(8*16-10, 6*16-12, bias=False)
#         self.fc3 = nn.Linear(6*16-12, 4*16-7, bias=False)
#         self.fc4 = nn.Linear(4*16-7, 2*16, bias=False)
        

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         # x = self.fc1(x)
#         # x = self.fc2(x)
#         # x = self.fc3(x)

#         #x = self.fc4(x)
#         return x

model = ModelClass()
# model.eval()
# input = torch.randn(4,2*16)  #(batch size, in-D)
# output = model(input)
#print(output)
from torch.fx import symbolic_trace
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)

# High-level intermediate representation (IR) - Graph representation
# print(symbolic_traced.graph)
# print(symbolic_traced.code)

gm = torch.fx.symbolic_trace(model)


# #do several gradient update
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr = 0.01)

# iteration = 1000
# for i in range(iteration):
#     input  = torch.ones(64,2*16)*1  #(batch size, input-Dim)

#     out = model(input)
#     ground = torch.ones(64,2*16)*100  #(batch size, outnput-Dim)
#     loss = criterion(out, ground)
#     print("loss:", loss.data.item())

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

print("Model's state_dict:")
#for param_tensor in model.state_dict():
    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #print(param_tensor, "\t", model.state_dict()[param_tensor].dtype)