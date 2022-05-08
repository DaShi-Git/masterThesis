#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> matmul_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> matmul_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> matmul_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
 
  return matmul_cuda_forward(input, weights, bias);
}

std::vector<torch::Tensor> matmul_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);

  return matmul_cuda_backward(
      grad_h,
      grad_cell);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matmul_forward, "Matmul forward (CUDA)");
  m.def("backward", &matmul_backward, "Matmul backward (CUDA)");
}