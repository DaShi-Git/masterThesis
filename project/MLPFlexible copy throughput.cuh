#pragma once

// i saved this on 22.08, 这一版实现了flexble mm 多层， 在一个warp中, 保存用来对比32 batch size 和16 bs，这一版是32，稳定版32bs
//#include <vector>
// #include"arbitaryActivation.cuh"
// #include"arbitaryHiddenChannels.cuh"
//#include"hiddenStructure.cuh"

#include "renderer_commons.cuh"
#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <mma.h>


template <typename scalar_t>
__device__ __forceinline__ scalar_t sine_act(scalar_t z) {
  return hsin(z);//1.0 / (1.0 + exp(-z));
}

#ifndef BLOCK_SIZE 
#define BLOCK_SIZE 96 //TO_BE_FILLED_BY_THE_HOST
#endif
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ int getCurrentWeightIndex(int layer){
  if (layer == 0){
    return 0;
  }
  int output = 0;
  for (int i = 0; i < layer; i++){
    output = output + hiddenChannels[i]*hiddenChannels[i+1]*16*16;
  }
  return output;
}

__device__ int getCurrentBiasIndex(int layer){
  if (layer == 0){
    return 0;
  }
  int output = 0;
  for (int i = 0; i < layer; i++){
    output = output + hiddenChannels[i+1]*16;
  }
  return output;
}

// // __device__	half d[32 * 32 *8];
 __device__	half a[128 * 128 *13];
//__device__	half a[12288];
__device__	half c[128 * 12];
// //__device__	half d[32 * 16 * 8*3*2];


__global__ void EvaluateMLPFlexible(
  //cudaTextureObject_t texObj, 
  kernel::Tensor2RW<real_t> test, kernel::Tensor2RW<real_t> weights, kernel::Tensor2RW<real_t> input, kernel::Tensor2RW<real_t> bias, kernel::Tensor2RW<real_t> output, int batchsizeTotal, int featuresize) {
    using namespace nvcuda;
      
// half a[128 * 128 *9];
// //__device__	half a[12288];
// half c[128 * 12];

__shared__	half d[32 * 16 * 8*3*2];
//__shared__	half d[12288];
//__shared__	half a[12288];
for(int i = threadIdx.x + blockDim.x*blockIdx.x; i < weights.size(1); i += blockDim.x*gridDim.x){
  a[i] = __float2half(weights[0][i]);
}
__syncthreads();
for(int i = threadIdx.x + blockDim.x*blockIdx.x; i < bias.size(1); i += blockDim.x*gridDim.x){
  c[i] = __float2half(bias[0][i]);
}
__syncthreads();


const int warpID = threadIdx.x / 32;
const int lineID = threadIdx.x % 32;
for (int j = 0; j < 100*5; ++j){
for(int i = threadIdx.x + blockDim.x*blockIdx.x; i < 20000; i += blockDim.x*gridDim.x){
  d[i] = a[i];
}
}
// int skew = 2;
// for (int j = 0; j < 100000*5; ++j){
// for(int i = (threadIdx.x + blockDim.x*blockIdx.x)*skew; i < 20000; i += (blockDim.x*gridDim.x)*skew){
//   *(int*)&d[i] = *(int*)&a[i];
//   //*(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];24576
// }
// }



 //end for hidden layers
__syncthreads();



// for(int i_output = 0; blockDim.x * i_output < batch_size*Cout16*16 * warpNum; ++i_output){//
//     int idx = blockDim.x * i_output + threadIdx.x;
//     //if(idx < batch_size*Cout16*16 * warpNum){
//        //output[0][idx] = __half2float(d[idx]);
//        //output[0][idx] = __half2float((half(8.0)));
//        //printf("thread %d < %d \n", idx, batch_size*Cout16*16 * warpNum);
//        //printf("idx %d", idx + loop * gridDim.x*Cout16*16*batch_size * warpNum + blockIdx.x*warpNum*Cout16*16*batch_size);
//         output[0][idx + loop * gridDim.x*Cout16*16*batch_size * warpNum + blockIdx.x*warpNum*Cout16*16*batch_size] = __half2float(d[idx]);
        
//     //}
// }
//end of if

 //end of loop grid
// printf("thread %d", threadIdx.x);

if (blockIdx.x == 0 && warpID==0 && threadIdx.x == 0){
  //printf("cout %d loop %d thread %d\n", Cout16, loop, threadIdx.x);
      for (int i = 0; i < output.size(0); i++) {
        for (int j = 0; j < output.size(1); j++) {
          //output[i][j] = __half2float(d[i+j*output.size(0)]);
          //output[0][j+i*output.size(0)] = __half2float(d[j+i*output.size(0)]);
        }
      }

      for (int i = 0; i < 1; ++i){
        //output[0][i] = __half2float(d[i]);
        // sWeightsHidden[i] = __(d[i]);
        // if (i % 32  == 0){ printf("\n");}
        // if (i % 32  == 0){ printf("\n");}
        printf("d[%d], %f ", i, __half2float(d[i]));
        printf("a[%d], %f ", i, __half2float(a[i]));

        //printf("bias[%d], %.3f", i, bias[0][i]);
        //printf("bias[%d], %d", i, bias.size(1));
      //   int out = (int)hiddenStructure[0];
      //printf("test1[0], %f", __half2float(test[0][0]));
      //printf("test1[0], %f", test[0][0]);
      // printf("a[%d], %.3f ", i, __half2float(b[i]));
      
      
        
      }
}
__syncthreads();


//     if (warpID==0 && blockIdx.x ==0){
//   printf("\n");
//   //printf("test value: %f", test[0][0]);
//   printf("input size: %d", input.size(1));
//   //printf("texture value is %f", tex2D<float>(texObj, 3, 0));
//   // half tmphalf = b1[3];
//   // printf("a1 value is %f", __half2float(tmphalf));
//   //printf("a1 value is %f", d1[0]);
// for (int p = 0; p < 32; p++)
// 						{
//   if (threadIdx.x == 32*warpID+p){
      
//       // printf("fragment b is %.3f", __half2float(b_frag[7][0].x[3]));
//       // printf("a[4000] is %.3f", __half2float(a[300]));
//       // printf("fragment lenght is %d", a_frag[1][0].num_elements);
//       //printf("thread [%d]" ,threadIdx.x);
//       //test_frag[0][0].x[1] = b_frag[0][0].x[3];
//       for (int t = 0; t < 8; t++)
// 						{
              
//               //if (test_frag[0][0].x[t] == (half)0){
//                 //if (t < 8){
// 							//printf("thread[%d] frag_c.x[%d] is %.3f \n",p,t, __half2float(c_frag[0][0].x[t]));
//               //printf("thread[%d] frag_b.x[%d] is %.3f \n",p,t, __half2float(test_frag[8][7].x[t]));
// 						 }
//             //}
//             //printf("\n");
//   }
//             }

//   }
  
    
}




    

