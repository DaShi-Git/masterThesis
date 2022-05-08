import torch
#import lltm
from matmul import Matmul #both import work from importing lltm nn model from lltm.py, see lines below
import time

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

batch_size = 16
input_features = 32
state_size = 128

# Note the device=cuda_device arguments here
X = torch.randn(4,4, device=cuda_device)
h = torch.randn(4,4, device=cuda_device)
C = torch.randn(4,4, device=cuda_device)

#rnn = lltm.LLTM(input_features, state_size).to(cuda_device)
rnn = Matmul(4,4).to(cuda_device)

forward = 0
backward = 0
for i in range(2):
    start = time.time()
    output = rnn(X, h, C)
    torch.cuda.synchronize()
    forward += time.time() - start

    start = time.time()
    # (new_h.sum() + new_C.sum()).backward()
    torch.cuda.synchronize()
    backward += time.time() - start
    print(i)
    print(X)
    print(output)

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))