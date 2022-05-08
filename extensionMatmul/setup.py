from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matmul_cuda',
    ext_modules=[
        CUDAExtension('matmul_cuda', [
            'matmul_cuda.cpp',
            'matmul_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })