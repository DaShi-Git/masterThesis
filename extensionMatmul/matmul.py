import math
from torch import nn
from torch.autograd import Function
import torch

#import lltm_cuda
import matmul_cuda

torch.manual_seed(42)


class MatmulFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        output = matmul_cuda.forward(input, weights, bias)
        #new_h, new_cell = outputs[:2]
        #variables = output[1:] + [weights]
        ctx.save_for_backward(input, weights, bias)

        return output[0]

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        output = matmul_cuda.backward(grad_h, grad_cell)
        #     grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        # d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return output[0]#d_input, d_weights, d_bias, d_old_h, d_old_cell


class Matmul(nn.Module):
    def __init__(self, input_features, state_size):
        super(Matmul, self).__init__()
        # self.input = nn.Parameter(torch.Tensor(4,4))
        # self.weights = nn.Parameter(
        #     torch.Tensor(4,4))
        # self.bias = nn.Parameter(torch.Tensor(4))
        

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.state_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, weights, bias):
        return MatmulFunction.apply(input, weights, bias)