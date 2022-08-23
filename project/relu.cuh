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

#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <mma.h>
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
#define BLOCK_SIZE 256 //TO_BE_FILLED_BY_THE_HOST
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



__shared__ alignas(32) half sWeightsHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS * HIDDEN_CHANNELS)];
__shared__ alignas(32) half sBiasHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS)];
__shared__ alignas(32) half sIntermediateResults[BLOCK_SIZE * HIDDEN_CHANNELS];

__global__ void EvaluateNoBatches2(
  kernel::Tensor2RW<real_t> positionsInput,
	kernel::Tensor2RW<real_t> densitiesOutput) {
    //half positions = __double2half(positionsInput)
    //warp and line ID
    assert(blockDim.x == BLOCK_SIZE);
    const int warpID = threadIdx.x / 32;
    const int lineID = threadIdx.x % 32;
    //printf(blockIdx.y);
    if (blockIdx.x == 0 && threadIdx.x == 0){
      // for (int i  = 0; i<positionsInput.size(0); ++i){
      //   printf("7");
      // };
      // for (int i  = 0; i<positionsInput.size(1); ++i){
      //   printf("2");
      // };
      printf("evaluta arbiacti, %f", arbiacti(-0.8));
      printf("blockDim.y, %d",blockDim.y);
      printf("threadIdx.y, %d",threadIdx.y);
      for (int j = 0; j<HIDDEN_CHANNELS; ++j){
          for (int b = 0; b<HIDDEN_CHANNELS; ++b){
            sWeightsHidden[j*HIDDEN_CHANNELS+b] = __float2half(positionsInput[j][b]);
          }
        }
      for (int i = 0; i<HIDDEN_CHANNELS*HIDDEN_CHANNELS;  ++i){
        //sWeightsHidden[i] = __float2half(positionsInput[i][0]);
        if (i<NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS){sBiasHidden[i] = __float2half(0.0f);}
        sIntermediateResults[i] = i==255? __float2half(1.0f):__float2half(1.0f);
        for (int j = 0; j<HIDDEN_CHANNELS; ++j){
          for (int b = 0; b<HIDDEN_CHANNELS; ++b){
            if (j==b){
              sIntermediateResults[j*HIDDEN_CHANNELS+b] = __float2half(1.0f);
            }
          }
        }
        //sIntermediateResults[i] = __float2half(densitiesOutput[i][0]);
        
        // printf("__double2half(positionsInput[i][0]), %d",__double2half(positionsInput[i][0]));
        // printf("positionsInput[i][0], %d",positionsInput[i][0]);
      }
      
      //printf("sWeightsHidden[0], %.3f",__half2float(sWeightsHidden[0]));
    };
    //storage for the intermediate results
		//half* intermediateResults = sIntermediateResults + 32 * HIDDEN_CHANNELS * warpID;

  for (int hidden = 0; hidden < NUM_HIDDEN_LAYERS; ++hidden)
			{
    static constexpr int Cin16 = HIDDEN_CHANNELS_DIV16;
    static constexpr int Cout16 = HIDDEN_CHANNELS_DIV16;

    //Cout = Cin = Batch=32
    using namespace nvcuda;
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    // Initialize the output to zero
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    // Load the inputs
    wmma::load_matrix_sync(a_frag, sWeightsHidden, 16);
    wmma::load_matrix_sync(b_frag, sIntermediateResults, 16);
    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // Store the output
    wmma::store_matrix_sync(sIntermediateResults, c_frag, 16, wmma::mem_row_major);


    
    // fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[HIDDEN_CHANNELS_DIV16][HIDDEN_CHANNELS_DIV16]; //row,col
    // fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[HIDDEN_CHANNELS_DIV16][1];
    // fragment<accumulator, 16, 16, 16, half> c_frag[HIDDEN_CHANNELS_DIV16][1];

    // //load C (bias)
    // //fill_fragment(c_frag, 0.0f);
    // for (int cout = 0; cout < Cout16; ++cout)
    // {
    //   load_matrix_sync(c_frag[cout][0], sBiasHidden + HIDDEN_CHANNELS * hidden + 16 * cout, 0, mem_col_major);
    //   //load_matrix_sync(c_frag[cout][1], sBiasHidden + HIDDEN_CHANNELS * hidden + 16 * cout, 0, mem_col_major);
    // }

    // //load A (weights)
    // for (int cout = 0; cout < Cout16; ++cout)
    //   for (int cin = 0; cin < Cin16; ++cin)
    //     load_matrix_sync(a_frag[cout][cin],
    //       sWeightsHidden + HIDDEN_CHANNELS * HIDDEN_CHANNELS * hidden + 16 * cin + HIDDEN_CHANNELS * 16 * cout,
    //       HIDDEN_CHANNELS);

    // //load B (input)
    // for (int cin = 0; cin < Cin16; ++cin)
    // {
    //   load_matrix_sync(b_frag[cin][0], sIntermediateResults + 16 * cin, HIDDEN_CHANNELS);
    //   //load_matrix_sync(b_frag[cin][1], intermediateResults + 16 * cin + 16 * HIDDEN_CHANNELS, HIDDEN_CHANNELS);
    // }

    // //matmul
    // for (int i = 0; i < Cout16; ++i) {
    //   //for (int j = 0; j < 2; ++j) {
    //     for (int k = 0; k < Cin16; ++k) {
    //       // mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
    //       mma_sync(c_frag[i][0], a_frag[i][k], b_frag[k][0], c_frag[i][0]);
    //     }
    //   //}
    // }

    // //copy to shared
    // for (int cout = 0; cout < Cout16; ++cout)
    // {
    //   store_matrix_sync(sIntermediateResults + 16 * cout, c_frag[cout][0], HIDDEN_CHANNELS, mem_col_major);
    //   //store_matrix_sync(intermediateResults + 16 * cout + 16 * HIDDEN_CHANNELS, c_frag[cout][1], HIDDEN_CHANNELS, mem_col_major);
    // }

      }//hidden layers
    //print
    
    if (blockIdx.x == 0 && threadIdx.x == 0){
      for (int i = 0; i<HIDDEN_CHANNELS*HIDDEN_CHANNELS;  ++i){
        if (i%HIDDEN_CHANNELS==0){printf("\n");}
        printf("%.3f ", __half2float(sIntermediateResults[i]));
        // printf("sIntermediateResults[%d], %.3f",i, __half2float(sIntermediateResults[i]));
        
      }
    }

    
    //std::cout << "activation test " << arbiacti(0.3) << std::endl;
    
    

// printf("12233");
// printf("/n");
  //return new_h;
  // this is a std::vector<torch::Tensor>
}