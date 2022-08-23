import torch
import torch.nn as nn
import numpy as numpy
from torch.nn import functional as F

class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        #注意 出入维度转换
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

model = ModelClass()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #print(param_tensor, "\t", model.state_dict()[param_tensor].dtype)

model.eval()
input = torch.randn(4,64)  #(batch size, in-D)
output = model(input)
print(output)