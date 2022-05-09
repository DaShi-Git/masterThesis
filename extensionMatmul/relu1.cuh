template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t z) {
  return z>1.0? z:1.0;//1.0 / (1.0 + exp(-z));
}

printf("relu1")