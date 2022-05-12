template <typename scalar_t>
__device__ __forceinline__ scalar_t arbiacti0(scalar_t z) {
return z>0.0? z:7.0;//1.0 / (1.0 + exp(-z));
}
