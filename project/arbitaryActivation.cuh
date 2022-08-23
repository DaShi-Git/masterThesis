template <typename scalar_t>
__device__ __forceinline__ scalar_t arbiacti1(scalar_t z) {
return z>0.0? z*z:7.0;
}
