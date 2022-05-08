this is the implementation of a lltm kernel from the official tutorial in Pytorch.

run in extensionCpp repository:
python setup.py install

a new package will be installed into the current environment, here is lltm-cuda package.

Now use can:
import torch
import lltm-cuda
to see whether the new package is installed successfully

run 
python lltm_evaluation.py
to evaluate the performance of lltm on GPU.
Evaluation on CPU is in lltm.py