#include <torch/extension.h>
#include <cuda.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <cuMat/src/DevicePointer.h>
#include <cuMat/src/Macros.h>
//#include "imodule.h"
#include "helper_math.cuh"
//#include "transfer_function.h"
#include "pytorch_utils.h"
#include "renderer_commons.cuh"
#include "kernel_loader.h"
#include "renderer_tensor.cuh"

#include <cuda_fp16.h>
#include <assert.h>



#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 1
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 2
#define N_TILES 2
#define K_TILES 2

int M_GLOBAL = (M * M_TILES);
int N_GLOBAL = (N * N_TILES);
int K_GLOBAL = (K * K_TILES);
// #define M_GLOBAL (M * M_TILES)
// #define N_GLOBAL (N * N_TILES)
// #define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 16 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 16

//struct cudaTextureDesc texDesc;
//texture<float, 1, cudaReaModeElementType> texRef;

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)
// #include "helper_cuda.h"
// #include "helper_functions.h"
// CUDA forward declarations
// from .cu !!
// std::vector<torch::Tensor> matmul_cuda_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias);

// std::vector<torch::Tensor> matmul_cuda_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell);

// C++ interface
// //from bindings.cpp//
// //#include <torch/extension.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include <tinyformat.h>
// #include <cuMat/src/Context.h>
// //#include <glm/gtx/string_cast.hpp>
// #include <third-party/Eigen/Core> // in cuMat

// //#include <kernel_loader.h>
// //#include <module_registry.h>
// //#include <opengl_utils.h>

// #ifdef WIN32
// #ifndef NOMINMAX
// #define NOMINMAX 1
// #endif
// #include <Windows.h>
// #endif

// #ifndef TORCH_EXTENSION_NAME
// #define TORCH_EXTENSION_NAME pyrenderer
// #endif

namespace py = pybind11;
// //using namespace renderer;

// #include "cmrc.hpp"
// CMRC_DECLARE(kernels);
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// std::vector<torch::Tensor> matmul_forward( //attention std::vector
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
 
//   return matmul_cuda_forward(input, weights, bias);
// }

// std::vector<torch::Tensor> matmul_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell) {
//   CHECK_INPUT(grad_h);
//   CHECK_INPUT(grad_cell);

//   return matmul_cuda_backward(
//       grad_h,
//       grad_cell);
// }

// std::vector<torch::Tensor> evaluate( //attention std::vector
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
 
//   return matmul_cuda_forward(input, weights, bias);
// }

__host__ void init_host_matrices(half *a, half *b, half *c) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i * K_GLOBAL + j] = __float2half(1.0f);
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = __float2half(2.5f);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = __float2half(0.0f);
  }
}


__host__ void matMultiplyOnHost(half *A, half *B, float *C, float alpha,
                                float beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
      }

      C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}




torch::Tensor evaluate_flexible_MLP(
	torch::Tensor& test, torch::Tensor& positions, const torch::Tensor& direction, const torch::Tensor& bias, 
  //at::Half& positions, const at::Half& direction, const torch::Tensor& bias, 
  const torch::Tensor& hiddenStructure, const int batchsize, const int featuresize, const int outputdim0, const int outputdim2, const std::list <std::string>& activation)
{
  CUstream stream = c10::cuda::getCurrentCUDAStream();
  //printf("111111111， test tensor %f",test[0][0]);
  
  //printf("111111111， test tensor %f",__half2float(test[0][0]));

  printf("below is the user defined activation function\n");
  for (auto const &i: activation) {
        std::cout << i << std::endl;
    }

half testhalf = (half)0.0;
printf("test half value is %f", __half2float(testhalf));
// start texture memory
const int height = 24;
    const int width = 24;
    float angle = 0.5;

    // Allocate and set some host data
    float *h_data = (float *)std::malloc(sizeof(float) * width * height);
    for (int i = 0; i < height * width; ++i)
        h_data[i] = i;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(float);
    // Copy data located at address h_data in host memory to device memory
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float),
                        height, cudaMemcpyHostToDevice);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    //texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  //end texture memory
half *source;
source = (half *)malloc(sizeof(half)*256);
for (int i=0; i<256; i++){
  source[i]=(half)18.0;

}
int sourceSize = 256 * sizeof(half);
half* d_data = NULL;
cudaMalloc((void**)&d_data, sourceSize);
cudaMemcpy(d_data, source, sourceSize, cudaMemcpyHostToDevice);
  //start normal cudaarray

  //end normal cudaarray

	// CHECK_CUDA(positions, true);
	// CHECK_DIM(positions, 2);
	// CHECK_SIZE(positions, 1, 3);
	bool hasDirection = false;
	// if (direction.defined())
	// {
	// 	hasDirection = true;
	// 	CHECK_CUDA(direction, true);
	// 	CHECK_DIM(direction, 2);
	// 	CHECK_SIZE(direction, 1, 3);
	// }

  //hidden structure
//   printf("hiddenlayer1, %d", hiddenStructure[0]);
//   printf("batchsize, %d", batchsize);
//   int *_hiddenStructure_h = NULL;

//   _hiddenStructure_h = (int *)malloc(sizeof(int) * hiddenStructure.size());
//   int *_hiddenStructure = NULL;

//   cudaMallocManaged((void **)&_hiddenStructure, sizeof(int) * hiddenStructure.size());
// for (int t = 0; t < hiddenStructure.size(); t++) {
//     _hiddenStructure_h[t] = hiddenStructure[t];
//   }

//   cudaMemcpy(_hiddenStructure, _hiddenStructure_h, sizeof(int) * hiddenStructure.size(),
//                              cudaMemcpyHostToDevice);
// printf("hiddenstructure0, %d", _hiddenStructure[0]);
// printf("hiddenstructure2, %d", _hiddenStructure[2]);
	//GlobalSettings s{};
	auto scalarType = positions.scalar_type();
	// s.volumeShouldProvideNormals = false;
	// s.interpolationInObjectSpace = false;
	// const auto oldBoxMax = boxMax();
	// const auto oldBoxMin = boxMin();
	// setBoxMin(make_double3(0, 0, 0));
	// setBoxMax(make_double3(1, 1, 1));
	// int channels = outputChannels();

	//kernel
	// this->prepareRendering(s);
	const std::string kernelName = "EvaluateMLPFlexible";
	std::vector<std::string> constantNames;
	// if (const auto c = getConstantDeclarationName(s); !c.empty())
	// 	constantNames.push_back(c);
	std::stringstream extraSource;
  for (auto const &i: activation) {
        extraSource << i << "\n";
    }
	extraSource << "#define MAX_N 4"
		//<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
    extraSource << "#define fragM 8"
		//<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
    extraSource << "#define frag_N 8"
		//<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
	// extraSource << "#define KERNEL_SYNCHRONIZED_TRACING "
	// 	<< (s.synchronizedThreads ? 1 : 0)
	// 	<< "\n";
	// extraSource << getDefines(s) << "\n";
	// fillExtraSourceCode(s, extraSource);
	// for (const auto& i : getIncludeFileNames(s))
	// 	extraSource << "\n#include \"" << i << "\"\n";
	// extraSource << "#define VOLUME_INTERPOLATION_T " <<
	// 	getPerThreadType(s) << "\n";
	//extraSource << "printf(\"relu1\")" << "\n";
	// extraSource << "#define VOLUME_USE_DIRECTION " << (hasDirection ? 1 : 0) << "\n";
	extraSource << "#include \"MLPFlexible.cuh\"\n";
	const auto fun0 = renderer::KernelLoader::Instance().getKernelFunction(
		kernelName, extraSource.str(), constantNames, false, false);
	if (!fun0.has_value())
		throw std::runtime_error("Unable to compile kernel");
	const auto fun = fun0.value();
	// if (auto c = getConstantDeclarationName(s); !c.empty())
	// {
	// 	CUdeviceptr ptr = fun.constant(c);
	// 	fillConstantMemory(s, ptr, stream);
	// }

	//output tensors
	int batches = positions.size(0);
  int channels = positions.size(1);
	auto densities = torch::empty({ outputdim0, outputdim2},
		at::TensorOptions().dtype(scalarType).device(c10::kCUDA));

// printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
//   printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
//   printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  half *A_h = NULL;
  half *B_h = NULL;
  half *C_h = NULL;
#if CPU_DEBUG
  float *result_hD = NULL;
  float *result_host = NULL;
#endif

  A_h = (half *)malloc(sizeof(half) * 64 * 64);
  B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
  C_h = (half *)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
  result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  result_host = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

  half *A = NULL;
  half *B = NULL;
  half *C = NULL;
  float *D = NULL;

//   cudaMalloc(reinterpret_cast<void **>(&A),
//                              sizeof(half) * M_GLOBAL * K_GLOBAL);
//   cudaMalloc(reinterpret_cast<void **>(&B),
//                              sizeof(half) * N_GLOBAL * K_GLOBAL);
//   cudaMalloc(reinterpret_cast<void **>(&C),
//                              sizeof(float) * M_GLOBAL * N_GLOBAL);
//   cudaMalloc(reinterpret_cast<void **>(&D),
//                              sizeof(float) * M_GLOBAL * N_GLOBAL);
cudaMallocManaged((void **)&A, sizeof(half) * 64 * 64);
cudaMallocManaged((void **)&B, sizeof(half) * N_GLOBAL * K_GLOBAL);
cudaMallocManaged((void **)&C, sizeof(half) * M_GLOBAL * N_GLOBAL);
cudaMallocManaged((void **)&D, sizeof(float) * M_GLOBAL * N_GLOBAL);

//   assert(((unsigned long long)A) % 128 == 0);
//   assert(((unsigned long long)B) % 128 == 0);
//   assert(((unsigned long long)C) % 128 == 0);
//   assert(((unsigned long long)D) % 128 == 0);

  
  init_host_matrices(A_h, B_h, C_h);
  // for (int i = 0; i < 64; i++) {
  //   for (int j = 0; j < 64; j++) {
  //     A_h[i * 64 + j] = test[i * 64 + j][0];
  //   }
  // }

  //printf("Preparing data for GPU...\n");

  cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL,
                             cudaMemcpyHostToDevice);
  cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL,
                             cudaMemcpyHostToDevice);
  cudaMemcpy(C, C_h, sizeof(half) * M_GLOBAL * N_GLOBAL,
                             cudaMemcpyHostToDevice);
  cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);


//start L2 cache
// cudaStream_t stream;
// cudaStreamCreate(&stream);                                                                  // Create CUDA stream

// cudaDeviceProp prop;                                                                        // CUDA device properties variable
// //cudaGetDeviceProperties( &prop, device_id);                                                 // Query GPU properties
// size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
// cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

int num_bytes = sizeof(half) * M_GLOBAL * K_GLOBAL*2;
//int num_bytes = sizeof(half) *256;
size_t window_size = num_bytes;
//size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(B);               // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                                        // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // Type of access property on cache miss

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream



  // enum {
  //   // Compute the right amount of shared memory to request.
  //   // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
  //   // per-CTA chunks
  //   // of the A and B matrices. Therefore, the right amount to request is the
  //   // maximum of those
  //   // two numbers.
  //   SHMEM_SZ = MAX(
  //       sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
  //       M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
  //           (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
  // };

  //printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  float alpha = 1.0f;
  float beta = 1.0f;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // If enough shared memory available on the GPU use high performant kernel
  //if (1 < 0) {
    // printf("Computing... using high performance kernel compute_gemm \n");

    // checkCudaErrors(cudaFuncSetAttribute(
    //     compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
    // checkKernelErrors(
    //     (compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
    //                     SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
#if CPU_DEBUG
    cudaMemcpy(result_hD, D,
                               sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyDeviceToHost);
#endif
  //} else {
    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    // blockDim.x = 128;
    // blockDim.y = 4;
    blockDim.x = 64;//96or256
    blockDim.y = 1;

    // gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
    //             (WMMA_M * blockDim.x / 32);
    // gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    gridDim.x = 3;
    gridDim.y = 1;
    printf("gridDim.x, %d, gridDim.y, %d", gridDim.x, gridDim.y);
for(int i = 0; i<4; ++i){
      //test1[i] = hiddenStructure_notuse[i];
      //printf("hiddenstructure[i], %d ", int(hiddenStructure[i]));
    }


    //printf("Computing... using simple_wmma_gemm kernel\n");
    // simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL,
    //                                         K_GLOBAL, alpha, beta);

  //}


	//launch kernel
	int blockSize;



	// {
		blockSize = 96;//96or256;   //fun.bestBlockSize();
	// }
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(batches, blockSize)),
		fun.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(batches), 1, 1 };
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(scalarType, "IVolumeInterpolation::evaluate", [&]()
		{
      const auto accHiddenStructure = accessor< ::kernel::Tensor1Read<scalar_t>>(hiddenStructure);
			//const auto accPosition = accessor< ::kernel::Tensor2Read<scalar_t>>(positions);
      const auto accPosition = accessor< ::kernel::Tensor2RW<scalar_t>>(positions);
      //const auto accPosition = positions;
			// const auto accDirection = hasDirection
			// 	? accessor< ::kernel::Tensor2Read<scalar_t>>(direction)
			// 	: ::kernel::Tensor2Read<scalar_t>();
      const auto accDirection = accessor< ::kernel::Tensor2Read<scalar_t>>(direction);
      const auto accBias = accessor< ::kernel::Tensor2Read<scalar_t>>(bias);
			const auto accDensity = accessor< ::kernel::Tensor2RW<scalar_t>>(densities);

      const auto accTest = accessor< ::kernel::Tensor2Read<scalar_t>>(test);
      // const auto accDirection = direction;
			// const auto accDensity = densities;
			const void *args[] = {&texObj, &accTest, &accPosition, &accDirection, &accBias, &accDensity, &A, &B, &C, &D, &M_GLOBAL, &N_GLOBAL,
                                            &K_GLOBAL, &alpha, &beta, &hiddenStructure, &batchsize, &featuresize};
			auto result = cuLaunchKernel(
				fun.fun(), gridDim.x, gridDim.y, 1, blockDim.x, blockDim.y, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return renderer::printError(result, kernelName);
			return true;
		});

	// setBoxMin(oldBoxMin);
	// setBoxMax(oldBoxMax);
	
	if (!success) throw std::runtime_error("Error during rendering!");

	#if CPU_DEBUG
    cudaMemcpy(result_hD, D,
                               sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyDeviceToHost);
	#endif
	cudaEventRecord(stop);
  cudaEventSynchronize(stop);

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

  matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
                    K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);
					for (int i = 0; i < 64; ++i){
        //sWeightsHidden[i] = __float2half(d[i]);
        //printf("result_host[%d], %.3f ", i, result_host[i]);
      //printf("a[%d], %.3f ", i, __half2float(b[i]));
        if (i % 16  == 0){ printf("\n");}
      }

//   for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
//     if (fabs(result_hD[i] - result_host[i]) > 0.1f)
//       printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i],
//              result_host[i]);
  //}
  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) *
                                                N_GLOBAL * K_GLOBAL * 2) /
                                               (milliseconds / 1000.)) /
                               1e12);

  free(A_h);
  free(B_h);
  free(C_h);
  cudaFree(reinterpret_cast<void *>(A));
  cudaFree(reinterpret_cast<void *>(B));
  cudaFree(reinterpret_cast<void *>(C));
  cudaFree(reinterpret_cast<void *>(D));
int deviceCount;
    // cudaGetDeviceCount(&deviceCount);
    // for(int i=0;i<deviceCount;i++)
    // {
    //     cudaDeviceProp deviceProp;
    //     cudaGetDeviceProperties(&deviceProp, i);
    //     // std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
    //     // std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
    //     // std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    //     // std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    //     // std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    //     // std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
    //     // std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    //     // std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    //     // std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
    //     //std::cout << "L2最大窗口： " << cudaDeviceProp::accessPolicyMaxWindowSize << std::endl;
    //     printf("  GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n",
    //         deviceProp.clockRate*1e-3f,deviceProp.clockRate*1e-6f);
    // printf("  Memory Bus width:                             %d-bits\n",
    //         deviceProp.memoryBusWidth);
    // if (deviceProp.l2CacheSize)
    // {
    //     printf("  L2 Cache Size:                            	%d bytes\n",
    //             deviceProp.l2CacheSize);
    // }
    // printf("  Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
    //         deviceProp.maxTexture1D,deviceProp.maxTexture2D[0],deviceProp.maxTexture2D[1]
    //         ,deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
    // printf("  Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n",
    //         deviceProp.maxTexture1DLayered[0],deviceProp.maxTexture1DLayered[1],
    //         deviceProp.maxTexture2DLayered[0],deviceProp.maxTexture2DLayered[1],
    //         deviceProp.maxTexture2DLayered[2]);
    // printf("  Total amount of constant memory               %lu bytes\n",
    //         deviceProp.totalConstMem);
    // printf("  Total amount of shared memory per block:      %lu bytes\n",
    //         deviceProp.sharedMemPerBlock);
    // printf("  Total number of registers available per block:%d\n",
    //         deviceProp.regsPerBlock);
    // printf("  Wrap size:                                    %d\n",deviceProp.warpSize);
    // printf("  Maximun number of thread per multiprocesser:  %d\n",
    //         deviceProp.maxThreadsPerMultiProcessor);
    // printf("  Maximun number of thread per block:           %d\n",
    //         deviceProp.maxThreadsPerBlock);
    // printf("  Maximun size of each dimension of a block:    %d x %d x %d\n",
    //         deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    // printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n",
    //         deviceProp.maxGridSize[0],
	  //   deviceProp.maxGridSize[1],
	  //   deviceProp.maxGridSize[2]);
    // printf("  Maximu memory pitch                           %lu bytes\n",deviceProp.memPitch);
    //     std::cout << "======================================================" << std::endl;     
        
    // }



// Destroy texture object
    cudaDestroyTextureObject(texObj);

    // Free device memory
    cudaFreeArray(cuArray);
    //cudaFree(output);

    // Free host memory
    free(h_data);
	
//global
cudaFree(d_data);
free(source);

//L2 cache
stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2 

	return densities;
}

// static void staticCudaSourcesLoaderRec(
// 	std::vector<renderer::KernelLoader::NameAndContent>& fileList,
// 	const cmrc::directory_entry& e, const cmrc::embedded_filesystem& fs,
// 	const std::string& currentPath)
// {
// 	if (e.is_file())
// 	{
// 		std::cout << "Load file " << e.filename() << std::endl;
// 		auto f = fs.open(currentPath + e.filename());
// 		std::string content(f.size(), '\0');
// 		memcpy(content.data(), f.begin(), f.size());
// 		fileList.push_back({ e.filename(), content });
// 	} else
// 	{
// 		std::cout << "Walk directory " << currentPath << std::endl;
// 		for (const auto& e2 : fs.iterate_directory(currentPath + e.filename()))
// 			staticCudaSourcesLoaderRec(fileList, e2, fs, currentPath + e.filename() + "/");
// 	}
// }

// static void staticCudaSourcesLoader(
// 	std::vector<renderer::KernelLoader::NameAndContent>& fileList)
// {
// 	cmrc::embedded_filesystem fs = cmrc::kernels::get_filesystem();
// 	for (const auto& e : fs.iterate_directory(""))
// 		staticCudaSourcesLoaderRec(fileList, e, fs, "");
// }

// std::filesystem::path getCacheDir()
// {
// 	//suffix and default (if default, it is a relative path)
// 	static const std::filesystem::path SUFFIX{ "kernel_cache" };
// #ifdef WIN32
// 	//get the path to this dll as base path
// 	char path[MAX_PATH];
// 	HMODULE hm = NULL;

// 	if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
// 		GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
// 		(LPCSTR)&getCacheDir, &hm) == 0)
// 	{
// 		int ret = GetLastError();
// 		fprintf(stderr, "GetModuleHandle failed, error = %d\n", ret);
// 		return SUFFIX;
// 	}
// 	if (GetModuleFileName(hm, path, sizeof(path)) == 0)
// 	{
// 		int ret = GetLastError();
// 		fprintf(stderr, "GetModuleFileName failed, error = %d\n", ret);
// 		return SUFFIX;
// 	}

// 	std::filesystem::path out = path;
// 	out = out.parent_path();
// 	const auto out_str = out.string();
// 	fprintf(stdout, "This DLL is located at %s, use that as cache dir\n", out_str.c_str());
// 	out /= SUFFIX;
// 	return out;
	
// #else
// 	return SUFFIX; //default
// #endif
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   renderer::KernelLoader::Instance().initCuda();
// 	renderer::KernelLoader::Instance().setCudaCacheDir(getCacheDir());
// 	renderer::KernelLoader::Instance().setCustomCudaSourcesLoader(staticCudaSourcesLoader);
// 	cuMat::Context& ctx = cuMat::Context::current();
// 	((void)ctx);
  // m.def("forward", &matmul_forward, "Matmul forward (CUDA)");
  // m.def("backward", &matmul_backward, "Matmul backward (CUDA)");
  m.def("evaluate_flexible_MLP", &evaluate_flexible_MLP, "Flexible MLP evaluate (CUDA)");
  auto cleanup_callback = []() {
		renderer::KernelLoader::Instance().cleanup();
#if RENDERER_OPENGL_SUPPORT==1
		OffscreenContext::teardown();
#endif
	};
	m.def("cleanup", cleanup_callback, py::doc("Explicit cleanup of all CUDA references"));
	m.add_object("_cleanup", py::capsule(cleanup_callback));
}