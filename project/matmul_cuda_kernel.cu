#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// namespace {
// template <typename scalar_t>
// __device__ __forceinline__ scalar_t haha(scalar_t z) {
//   return z*2.0; //z>1.0? z:1.0;//1.0 / (1.0 + exp(-z));
// }
 // namespace  
 // namespace must be ending here, otherwise error by importing matmul_cuda, undefined matmul_cuda_backword

__global__ void matmul_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
    //auto new_h = haha(input);
    input = input*3.0;
printf("12233");
printf("/n");
  //return new_h;
  // this is a std::vector<torch::Tensor>
}

std::vector<torch::Tensor> matmul_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell) {
    auto new_h = torch::zeros_like(grad_h);


  return {new_h};
}

