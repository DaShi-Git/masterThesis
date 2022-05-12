#pragma once
//#include <torch/extension.h>

// #include <cuda.h>
// #include <cuda_runtime.h>

//#include <vector>


#include "renderer_commons.cuh"
#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
// namespace {
// template <typename scalar_t>
// __device__ __forceinline__ scalar_t relu(scalar_t z) {
//   return z>0.0? z:0.0;//1.0 / (1.0 + exp(-z));
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

__global__ void EvaluateNoBatches2(
  kernel::Tensor2RW<real_t> positionsInput,
	kernel::Tensor2RW<real_t> densitiesOutput) {
    //auto new_h = haha(input);
    //densitiesOutput = positionsInput; //*3.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //printf(blockIdx.y);
    if (blockIdx.x == 0 && threadIdx.x == 0){
      for (int i  = 0; i<positionsInput.size(0); ++i){
        printf("3");
      };
      for (int i  = 0; i<positionsInput.size(1); ++i){
        printf("2");
      };
    };
    

// printf("12233");
// printf("/n");
  //return new_h;
  // this is a std::vector<torch::Tensor>
}