from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, _get_cuda_arch_flags, _join_cuda_home
import os

_arch_flags = _get_cuda_arch_flags()
print('arch flags:', _arch_flags)

_root = os.path.split(os.path.abspath(__file__))[0]
print("root path:", _root)

_common_args = [
    '-std=c++17',
    '-DNVCC_ARGS="%s"'%_arch_flags[0],
    '-DNVCC_INCLUDE_DIR=%s'%_join_cuda_home('include'),
    '-DUSE_DOUBLE_PRECISION=1',
    '-DRENDERER_OPENGL_SUPPORT=0',
    '-DCDUMAT_SINGLE_THREAD_CONTEXT=1',
    '-DTHRUST_IGNORE_CUB_VERSION_CHECK=1',
]

_include_dirs = [
#    '%s/renderer'%_root,
    '%s/third-party/cuMat'%_root,
    '%s/third-party/cuMat/third-party'%_root,
    '%s/third-party/magic_enum/include'%_root,
    # '%s/third-party/cudad/include/cudAD'%_root,
    '%s/third-party/tinyformat'%_root,
    # '%s/third-party/nlohmann'%_root,
    # '%s/third-party/lz4/lib'%_root,
    # '%s/third-party/portable-file-dialogs'%_root,
    # '%s/third-party/thread-pool/include'%_root,
    # '%s/build/_cmrc/include'%_root,
    # '%s/imgui'%_root,
    '/usr/include',
]

_libraries = [
    'cuda',
    'nvrtc',
    'curand',
    # 'GL', 'GLU',
]

setup(
    name='matmul_cuda',
    ext_modules=[
        CUDAExtension('matmul_cuda', [
            'bindings.cpp',
            # 'sample.cu',
            #'matmul_cuda_kernel.cu',
            'kernel_loader.cpp',
            'sha1.cpp'
        ],
        extra_compile_args = {
                'cxx': _common_args,
                'nvcc': _common_args+["--extended-lambda"]
            },
        include_dirs = _include_dirs,
        libraries = _libraries
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })