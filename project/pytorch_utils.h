#pragma once

#include <torch/types.h>

#define CHECK_CUDA(x, cuda) TORCH_CHECK(x.device().is_cuda()==cuda, "All tensors must be on the same device, but " #x " is different, is_cuda=", x.device().is_cuda())
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x, d)	\
	TORCH_CHECK(x.defined(), #x " is not defined!");	\
	TORCH_CHECK((x.dim() == (d)), #x " must be a tensor with ", d, " dimensions, but has shape ", x.sizes())
#define CHECK_SIZE(x, d, s) TORCH_CHECK((x.size(d) == (s)), #x " must have ", s, " entries at dimension ", d, ", but has ", x.size(d), " entries")
#define CHECK_DTYPE(x, t) TORCH_CHECK(x.dtype()==t, #x " must be of type ", t, ", but is ", x.dtype())
#define CHECK_MATCHING_DTYPE(x1, x2) TORCH_CHECK(x1.dtype()==x2.dtype(), \
    "Dtypes of the input tensors must be identical, but the dtype of " #x1 " is ", x1.dtype(), " and of " #x2 " is ", x2.dtype())
inline int checkBatch(const torch::Tensor& t, const char* name, int B1, int batchIndex = 0)
{
	int B2 = t.size(batchIndex);
	if (B2 == 1) return B1; //broadcasting to old batch size
	if (B1 == 1) return B2; //broadcasting to new batch size
	TORCH_CHECK(B1 == B2, name, " must be have the same batch size as other tensors, if batch>1. actual=", B2, ", expected=", B1);
	return B1;
}
#define CHECK_BATCH(x, b) checkBatch(x, #x, b)

inline caffe2::TypeMeta checkSupportedRealType(const torch::Tensor& t, const char* name)
{
	TORCH_CHECK(t.dtype() == c10::ScalarType::Float || t.dtype() == c10::ScalarType::Double,
		name, " must be of type 'float' or 'double', but is of type ", t.dtype());
	return t.dtype();
}
#define CHECK_REAL_TYPE(t) checkSupportedRealType(t, #t)

template<typename T>
inline c10::ScalarType getPytorchType();
template<>
inline c10::ScalarType getPytorchType<float>() { return c10::kFloat; }
template<>
inline c10::ScalarType getPytorchType<double>() { return c10::kDouble; }

template<typename Accessor>
Accessor accessor(const torch::Tensor& t)
{
	return Accessor(static_cast<typename Accessor::PtrType>(t.data_ptr<typename Accessor::Type>()), t.sizes().data(), t.strides().data());
}
