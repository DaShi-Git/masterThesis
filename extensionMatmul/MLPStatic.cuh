#pragma once
//#include <torch/extension.h>
//#include <iostream>
// #include <cuda.h>
// #include <cuda_runtime.h>

//#include <vector>
#include"arbitaryActivation.cuh"

#include "renderer_commons.cuh"
#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <mma.h>
//#include <stdio.h>
//#include "stdlib.h"
// namespace {
// template <typename scalar_t>
// __device__ __forceinline__ scalar_t arbiacti(scalar_t z) {
//   return z>0.5? z:0.5;//1.0 / (1.0 + exp(-z));
// }
 // namespace  
 // namespace must be ending here, otherwise error by importing matmul_cuda, undefined matmul_cuda_backword

// std::vector<torch::Tensor> matmul_cuda_forward(
//     torch::Tensor& input,
//     torch::Tensor& weights,
//     CUstream stream) {
//     auto new_h = relu(input);
// printf("123");
// printf("/n");
//   return {new_h};
//   // this is a std::vector<torch::Tensor>
// }

// std::vector<torch::Tensor> matmul_cuda_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell) {
//     auto new_h = torch::zeros_like(grad_h);


//   return {new_h};
// }
#define KERNEL_DOUBLE_PRECISION 0


#ifndef BLOCK_SIZE 
#define BLOCK_SIZE 512 //TO_BE_FILLED_BY_THE_HOST
#endif

//the number of hidden layers
#ifndef NUM_HIDDEN_LAYERS
#define NUM_HIDDEN_LAYERS 1 //TO_BE_FILLED_BY_THE_HOST
#endif

#ifndef HIDDEN_CHANNELS_DIV16
#define HIDDEN_CHANNELS_DIV16 1 // TO_BE_FILLED_BY_THE_HOST
#endif
#define HIDDEN_CHANNELS (16*(HIDDEN_CHANNELS_DIV16))


#define MIN_ONE(val) (((val)>0)?(val):(1))


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

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

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
// __shared__ alignas(32) half sWeightsHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS * HIDDEN_CHANNELS)];
// __shared__ alignas(32) half sBiasHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS)];
// __shared__ alignas(32) half sIntermediateResults[BLOCK_SIZE * HIDDEN_CHANNELS];
// __device__ void init_host_matrices(half a, half b, float c) {
//   for (int i = 0; i < M_GLOBAL; i++) {
//     for (int j = 0; j < K_GLOBAL; j++) {
//       a[i * K_GLOBAL + j] = __float2half(1.0f);
//     }
//   }

//   for (int i = 0; i < N_GLOBAL; i++) {
//     for (int j = 0; j < K_GLOBAL; j++) {
//       b[i * K_GLOBAL + j] = __float2half(1.0f);
//     }

//   for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
//     c[t] = 3.0f;
//   }
// }}


// Matrix on device


//__device__ void init_host_matrices(half a, half b, float c) {
//}
	// CUDA Unified Memory 
// cudaMallocManaged((void **)&a, sizeof(half) * M_GLOBAL * K_GLOBAL);
// cudaMallocManaged((void **)&b, sizeof(half) * N_GLOBAL * K_GLOBAL);
// cudaMallocManaged((void **)&c, sizeof(float) * M_GLOBAL * N_GLOBAL);
// cudaMallocManaged((void **)&d, sizeof(float) * M_GLOBAL * N_GLOBAL);
	
	// Init matrix A B C on host
	//InitHostMatrix(host_A, host_B, host_C);
	//printf("[*] Initializing Matrix...\n");
	//init_host_matrices(a, b, c);
	
	// computing gemm using tensor core
	//printf("[*] Computing D = A * B +C with Tensor Cores...\n");



//__shared__ float sD[M_GLOBAL * N_GLOBAL*12];
__global__ void EvaluateMLPStatic(
  half *a1, half *b1, float *c1, float *d1, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {
    using namespace nvcuda;
    // begin first layer
    __shared__	half d[M_GLOBAL * N_GLOBAL];
    for (int hidden = 0; hidden < 1; ++hidden){
    __shared__	half a[M_GLOBAL * K_GLOBAL];
    __shared__	half b[K_GLOBAL * N_GLOBAL];
    __shared__	half c[M_GLOBAL * N_GLOBAL];
    // __shared__	half d[M_GLOBAL * N_GLOBAL];
  // Leading dimensions. Packed with no transpositions.
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;
  if (blockIdx.x == 0 && threadIdx.x == 0){
for (int i = 0; i < M_GLOBAL; i++) {
  for (int j = 0; j < K_GLOBAL; j++) {
    a[i * K_GLOBAL + j] = __float2half(1.0f);
  }
}

for (int i = 0; i < N_GLOBAL; i++) {
  for (int j = 0; j < K_GLOBAL; j++) {
    b[i * K_GLOBAL + j] = __float2half(1.0f);
  }

for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
  c[t] = 0.0f;
}
}}
__syncthreads();
  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag;
  //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

  //wmma::fill_fragment(acc_frag, __float2half(0.0f));

  //load c
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                           wmma::mem_row_major);
  }

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * N;
    int bRow = i;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  if (cRow < m_ld && cCol < n_ld) {
    
    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                            wmma::mem_col_major);
  }
  
  if (warpM==0&&warpN==0 && threadIdx.x == 0){ // && threadIdx.x == 0){
      //printf("threadIdx.x, %d ", threadIdx.x);
      for (int i = 0; i < 256; ++i){
        //sWeightsHidden[i] = __float2half(d[i]);
        if (i % 16  == 0){ printf("\n");}
        //printf("d[%d], %.3f ", i, __half2float(d[i]));
        
      //printf("a[%d], %.3f ", i, __half2float(b[i]));
        
      }
    }
    __syncthreads();
 }
    // end first layer
//__syncwarp();
    // begin hidden layers
    //hiddenStructure = [1, 1, 1] 
  for (int hidden = 0; hidden < 3; ++hidden){
    __shared__	half a[M_GLOBAL * K_GLOBAL];
    //__shared__	half b[K_GLOBAL * N_GLOBAL];
    __shared__	half c[M_GLOBAL * N_GLOBAL];
    // __shared__	half d[M_GLOBAL * N_GLOBAL];
  // Leading dimensions. Packed with no transpositions.
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;
  if (blockIdx.x == 0 && threadIdx.x == 0){
for (int i = 0; i < M_GLOBAL; i++) {
  for (int j = 0; j < K_GLOBAL; j++) {
    a[i * K_GLOBAL + j] = __float2half(0.1f);
  }
}

for (int i = 0; i < N_GLOBAL; i++) {
  for (int j = 0; j < K_GLOBAL; j++) {
    //b[i * K_GLOBAL + j] = __float2half(1.0f);
    
  }

for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
  c[t] = 0.0f;
}
}}
  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  // if (warpM==0&&warpN==0 && threadIdx.x == 0){
      
  //     for (int i = 0; i < 256; ++i){
  //       //sWeightsHidden[i] = __float2half(d[i]);
  //       if (i % 16  == 0){ printf("\n");}
  //       printf("d[%d], %.3f ", i, __half2float(d[i]));
  //     //printf("a[%d], %.3f ", i, __half2float(b[i]));
        
  //     }
  //   }
__syncthreads();
  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag;
  //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

  //wmma::fill_fragment(acc_frag, __float2half(0.0f));

  //load c
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                           wmma::mem_row_major);
  }

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * N;
    int bRow = i;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, d + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  if (cRow < m_ld && cCol < n_ld) {
    
    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                            wmma::mem_col_major);
  }
  
//   if (warpM==0&&warpN==0 && threadIdx.x == 0){
      
//       for (int i = 0; i < 256; ++i){
//         //sWeightsHidden[i] = __float2half(d[i]);
//         if (i % 16  == 0){ printf("\n");}
//         printf("d[%d], %.3f ", i, __half2float(d[i]));
//       //printf("a[%d], %.3f ", i, __half2float(b[i]));
        
//       }
//     }
// __syncthreads();

  }
  
  // __syncwarp();
    //end hidden layers
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  // __syncthreads();
    if (warpM==0&&warpN==0 && threadIdx.x == 0){
      
      for (int i = 0; i < 256; ++i){
        //sWeightsHidden[i] = __float2half(d[i]);
        if (i % 16  == 0){ printf("\n");}
        printf("d[%d], %.3f ", i, __half2float(d[i]));
      //printf("a[%d], %.3f ", i, __half2float(b[i]));
        
      }
    }
    
}