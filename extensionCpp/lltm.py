import math
from torch import nn
from torch.autograd import Function
import torch

import lltm_cuda

torch.manual_seed(42)


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cuda.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cuda.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)

import time

#below is the performance evaluation on CPU

# batch_size = 16
# input_features = 32
# state_size = 128

# X = torch.randn(batch_size, input_features)
# h = torch.randn(batch_size, state_size)
# C = torch.randn(batch_size, state_size)

# rnn = LLTM(input_features, state_size)

# forward = 0
# backward = 0
# for _ in range(100000):
#     start = time.time()
#     new_h, new_C = rnn(X, (h, C))
#     forward += time.time() - start

#     start = time.time()
#     (new_h.sum() + new_C.sum()).backward()
#     backward += time.time() - start

# print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))

#below is the performance evaluation on GPU

# assert torch.cuda.is_available()
# cuda_device = torch.device("cuda")  # device object representing GPU

# batch_size = 16
# input_features = 32
# state_size = 128

# # Note the device=cuda_device arguments here
# X = torch.randn(batch_size, input_features, device=cuda_device)
# h = torch.randn(batch_size, state_size, device=cuda_device)
# C = torch.randn(batch_size, state_size, device=cuda_device)

# rnn = LLTM(input_features, state_size).to(cuda_device)

# forward = 0
# backward = 0
# for i in range(100000):
#     start = time.time()
#     new_h, new_C = rnn(X, (h, C))
#     torch.cuda.synchronize()
#     forward += time.time() - start

#     start = time.time()
#     (new_h.sum() + new_C.sum()).backward()
#     torch.cuda.synchronize()
#     backward += time.time() - start
#     print(i)

# print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

#result output: Forward: 455.091 us | Backward 931.705 us
#not so fast as in tutorial: Forward: 149.802 us | Backward 393.458 us