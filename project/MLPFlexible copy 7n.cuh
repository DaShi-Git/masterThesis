#pragma once
//#include <torch/extension.h>
//#include <iostream>
// #include <cuda.h>
// #include <cuda_runtime.h>

// problem by 16x16 there's -63488 error starting from index 129, 131, 133...


//#include <vector>
#include"arbitaryActivation.cuh"
#include"arbitaryHiddenChannels.cuh"
#include"hiddenStructure.cuh"

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
template <typename scalar_t>
__device__ __forceinline__ scalar_t sine_act(scalar_t z) {
  return hsin(z);//1.0 / (1.0 + exp(-z));
}
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

#define M_TILES 8
#define N_TILES 2
#define K_TILES 8

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
// const int hiddenChannels[4] = {2, 1, 2, 1};
__device__ int getCurrentWeightIndex(int layer, int batchsize, int featuresize){
  if (layer == 0){
    return 0;
  }
  int output = 0;
  int tmphiddenChannels[sizeof(hiddenChannels)+1];
  tmphiddenChannels[0] = featuresize;
  for (int i = 0; i < sizeof(hiddenChannels); i++){
    tmphiddenChannels[i+1] = hiddenChannels[i];
  }
  for (int i = 0; i < layer; i++){
    output = output + tmphiddenChannels[i]*tmphiddenChannels[i+1]*16*16;

  }
  return output;

}

// __device__	half d[32 * 32 *8];
// __device__	half a[32 * 32 *8];
// __device__	half c[32 * 32 *8];



//__shared__ float sD[M_GLOBAL * N_GLOBAL*12];
__global__ void EvaluateMLPFlexible(
  kernel::Tensor2RW<real_t> weights, kernel::Tensor2RW<real_t> input, kernel::Tensor2RW<real_t> bias, kernel::Tensor2RW<real_t> output, half *a1, half *b1, float *c1, float *d1, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta, kernel::Tensor1RW<real_t> hiddenStructure_notuse, int batchsize, int featuresize) {
    using namespace nvcuda;
    
    const int M_GLOBAL1 = (M * M_TILES);
    const int N_GLOBAL1 = (N * N_TILES);
    const int K_GLOBAL1 =  (K * K_TILES);
  

    __shared__	half d[M_GLOBAL1 * N_GLOBAL1]; // here need to be changed
    __shared__	half a[M_GLOBAL1 * K_GLOBAL1];
    //__shared__	half b[K_GLOBAL1 * N_GLOBAL1*6];
    __shared__	half c[M_GLOBAL1 * N_GLOBAL1];
    __shared__	half shbias[16*16];

    const int warpID = threadIdx.x / 32;
		const int lineID = threadIdx.x % 32;
    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
    //   new_frag[8][2];
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> new_frag[8][2];
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag[2][2];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag[2][2];
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[2][2];
    
    //for (int hidden = 0; hidden < 1; ++hidden){



    
    const int M_GLOBAL = 8*16;
    const int N_GLOBAL = 32;
    const int K_GLOBAL = 8*16;
  // Leading dimensions. Packed with no transpositions.
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;
  //int currentWeightIndex = getCurrentWeightIndex(hidden, batchsize, featuresize);
  int numloop_a = (M_GLOBAL*K_GLOBAL) / BLOCK_SIZE;
 if (((M_GLOBAL*K_GLOBAL) % BLOCK_SIZE)>0){
   numloop_a = numloop_a + 1;
 }


//   for (int i = 0; i < numloop_a; i++){
//     if (i * BLOCK_SIZE + threadIdx.x < M_GLOBAL*K_GLOBAL){ 
//     a[i * BLOCK_SIZE + threadIdx.x] = __float2half(weights[currentWeightIndex+i * BLOCK_SIZE + threadIdx.x][0]);
//     }

//   }

//   int numloop_d = (N_GLOBAL*K_GLOBAL) / BLOCK_SIZE;
//  if (((N_GLOBAL*K_GLOBAL) % BLOCK_SIZE)>0){
//    numloop_d = numloop_d + 1;
//  }
//   for (int i = 0; i < numloop_d; i++){
//     if (i * BLOCK_SIZE + threadIdx.x < N_GLOBAL*K_GLOBAL){ 
//     d[i * BLOCK_SIZE + threadIdx.x] = __float2half(input[i * BLOCK_SIZE + threadIdx.x][0]);
//     }

//   }
//__syncthreads();
//if (warpID==0){
  
  //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
  // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

static constexpr int Cin16 = 2;
static constexpr int Cout16 = 2;
  //load A (weights)
// #pragma unroll
// 				for (int cout = 0; cout < Cout16; ++cout){
// #pragma unroll
// 					for (int cin = 0; cin < Cin16; ++cin){
// 						wmma::load_matrix_sync(a_frag[cout][cin],
// 							a + 16 * cin + Cin16*16 * 16 * cout,
// 							Cin16*16);
//           }
//         }

        //load B (input)
//#pragma unroll
				// for (int cin = 0; cin < Cin16; ++cin)
				// {
				// 	wmma::load_matrix_sync(b_frag[cin][0], d + 16 * cin, Cin16*16);
				// 	wmma::load_matrix_sync(b_frag[cin][1], d + 16 * cin + 16 * Cin16*16, Cin16*16);
				// }

        for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < Cin16; ++j) {
						for (int t = 0; t < a_frag[0][0].num_elements; t++)
						{
              //half tmp = (half)1;//c_frag[i][j].x[t];
              // tmp = tmp + (half)5;
							// c_frag[i][j].x[t] = tmp;
              a_frag[i][j].x[t] = (half)1;
              //b_frag.x[t] = (half)1;//c_frag[i][j].x[t];
              //printf("c.x[%d] is %.3f",t, __half2float(tmp));
              
						}
					}
				}
        for (int i = 0; i < Cin16; ++i) {
					for (int j = 0; j < 2; ++j) {
						for (int t = 0; t < b_frag[0][0].num_elements; t++)
						{
              b_frag[i][j].x[t] = (half)1;

						}
					}
				}
        for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < 2; ++j) {
						for (int t = 0; t < c_frag[0][0].num_elements; t++)
						{
              c_frag[i][j].x[t] = (half)1;

						}
					}
				}
//__syncthreads();
// __syncthreads();
//__syncwarp();
				//matmul
        //#pragma unroll
				for (int i = 0; i < Cout16; ++i) {
          //#pragma unroll
					for (int j = 0; j < 2; ++j) {
           //#pragma unroll
						for (int k = 0; k < 2; ++k) {
							wmma::mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
              //__syncthreads();
						}
					}
				}
//         wmma::mma_sync(c_frag[0][0], a_frag[0][0], b_frag[0][0], c_frag[0][0]);
        
//         wmma::mma_sync(c_frag[0][1], a_frag[0][0], b_frag[1][0], c_frag[0][1]);

//       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> tmp_frag; 
//       for (int t = 0; t < c_frag[0][0].num_elements; t++)
// 						{
//               tmp_frag.x[t]=c_frag[0][0].x[t];
// 						}
// //wmma::mma_sync(tmp_frag, a_frag[0][1], b_frag[1][0], tmp_frag);
//         wmma::mma_sync(c_frag[0][0], a_frag[0][1], b_frag[1][0], c_frag[0][0]);
       
        //__syncthreads();

        //activations
				// for (int i = 0; i < 1; ++i) {
				// 	for (int j = 0; j < 1; ++j) {
				// 		for (int t = 0; t < c_frag[0][0].num_elements; t++)
				// 		{
        //       half tmp = (half)1;//c_frag[i][j].x[t];
        //       // tmp = tmp + (half)5;
				// 			// c_frag[i][j].x[t] = tmp;
        //       //b_frag[i][j].x[t] = (half)1;//c_frag[i][j].x[t];
        //       //printf("c.x[%d] is %.3f",t, __half2float(tmp));
              
				// 		}
				// 	}
				// }
        


        //copy to shared new_frag
        half* intermediateResults = d + 32 * Cin16*16 * warpID;
        //#pragma unroll
				for (int cout = 0; cout < Cout16; ++cout)
				{
					// wmma::store_matrix_sync(intermediateResults + 16 * cout, c_frag[cout][0], Cin16*16, wmma::mem_col_major);
					// wmma::store_matrix_sync(intermediateResults + 16 * cout + 16 * Cin16*16, c_frag[cout][1], Cin16*16, wmma::mem_col_major);
				}
//__syncthreads();

// //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> test_frag[1][1];
// wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> test_frag[1][1];
//  if (threadIdx.x == 0){
//    for (int i=0; i<16*16; ++i){
//      shbias[i] = __float2half(bias[i][0]);
//    }

//  }
//  __syncthreads();
// //  wmma::load_matrix_sync(test_frag[0][0], shbias, 16, wmma::mem_col_major);
// //wmma::load_matrix_sync(test_frag[0][0], shbias, 16, wmma::mem_row_major);
// wmma::load_matrix_sync(test_frag[0][0], shbias, 16);



//   printf("\n");
// for (int p = 0; p < 32; p++)
// 						{
//   if (threadIdx.x == p){
      
//       // printf("fragment b is %.3f", __half2float(b_frag[7][0].x[3]));
//       // printf("a[4000] is %.3f", __half2float(a[300]));
//       // printf("fragment lenght is %d", a_frag[1][0].num_elements);
//       //printf("thread [%d]" ,threadIdx.x);
//       //test_frag[0][0].x[1] = b_frag[0][0].x[3];
//       for (int t = 0; t < c_frag[0][0].num_elements; t++)
// 						{
//               //if (test_frag[0][0].x[t] == (half)0){
//                 if (t < 16){
// 							//printf("thread[%d] frag_b.x[%d] is %.3f \n",p,t, __half2float(c_frag[0][1].x[t]));
// 						 }
//             }
//             printf("\n");
//   }
//}
//} //if warp=0
//__syncthreads();
//__syncwarp();

  
  //__syncthreads();
    //end hidden layers
    int warpM = warpID / MAX_N;
  int warpN = warpID % MAX_N;
  // __syncthreads();
    // if (warpM==0&&warpN==0 && threadIdx.x == 0){
    //   printf("arbiact %f", sine_act(2.0));
    //   printf("sizeofa %d", sizeof(half));
    //   //printf("fragment a is %f", __half2float(a_frag[2][1].x[3]));
    //   for (int i = 0; i < output.size(0); i++) {
    //     printf("\n");
    //     for (int j = 0; j < output.size(1); j++) {
    //       output[i][j] = __half2float(d[i+j*output.size(0)]);
    //       //printf("d[%d][%d], %.3f ", i, j, __half2float(d[i+j*output.size(0)]));
    //     }
    //   }
      
    //   for (int i = 0; i < 16*16*2*8; ++i){
    //     //sWeightsHidden[i] = __float2half(d[i]);
    //     // if (i % 32  == 0){ printf("\n");}
    //     // if (i % 32  == 0){ printf("\n");}
    //     //printf("d[%d], %.3f ", i, __half2float(d[i]));
    //     //printf("a[%d], %.3f ", i, __half2float(a[i]));

    //     // printf("input[%d], %f", i, __half2float(b[i]));
    //     // int out = (int)hiddenStructure[0];
    //     //printf("test1[3], %d", test1[0]);
    //   //printf("a[%d], %.3f ", i, __half2float(b[i]));
      
      
        
    //   }
    // }
       //__syncthreads();
       
       //__syncwarp();
 // __syncthreads();
    if (warpID==0){
  printf("\n");
for (int p = 0; p < 32; p++)
						{
  if (threadIdx.x == p){
      
      // printf("fragment b is %.3f", __half2float(b_frag[7][0].x[3]));
      // printf("a[4000] is %.3f", __half2float(a[300]));
      // printf("fragment lenght is %d", a_frag[1][0].num_elements);
      //printf("thread [%d]" ,threadIdx.x);
      //test_frag[0][0].x[1] = b_frag[0][0].x[3];
      for (int t = 0; t < c_frag[0][0].num_elements; t++)
						{
              //if (test_frag[0][0].x[t] == (half)0){
                if (t < 16){
							printf("thread[%d] frag_b.x[%d] is %.3f \n",p,t, __half2float(c_frag[0][1].x[t]));
              //printf("thread[%d] frag_b.x[%d] is %.3f \n",p,t, __half2float(tmp_frag.x[t]));
						 }
            }
            printf("\n");
  }
            }

  }
//__syncthreads();
  //} //for hidden
   //__syncthreads();
  //__syncwarp();
  //__syncthreads();
  
    
}




    

