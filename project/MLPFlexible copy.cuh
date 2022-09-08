#pragma once

// i saved this on 22.08, 这一版实现了flexble mm 多层， 在一个warp中
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

// __device__	half d[32 * 32 *8];
__device__	half a[128 * 128 *2];
__device__	half c[128 * 2];


// __device__ half shbias[16*16];


//__shared__ float sD[M_GLOBAL * N_GLOBAL*12];
__global__ void EvaluateMLPFlexible(
  cudaTextureObject_t texObj, kernel::Tensor2Read<real_t> test, kernel::Tensor2RW<real_t> weights, kernel::Tensor2RW<real_t> input, kernel::Tensor2RW<real_t> bias, kernel::Tensor2RW<real_t> output, half *a1, half *b1, float *c1, float *d1, int batchsizeTotal, int featuresize) {
    using namespace nvcuda;
    
    // const int M_GLOBAL1 = (M * M_TILES);
    // const int N_GLOBAL1 = (N * N_TILES);
    // const int K_GLOBAL1 =  (K * K_TILES);

    // const int offset = 0;
    
  
__shared__	half d[32 * 16 * 8*6];

//set all c[] to 0
// if (threadIdx.x == 0){


//       for (int i = 0; i < 128 * 2; ++i){
//         c[i] = __float2half(0.0f);

      
        
//       }
// }


// __syncthreads();

    const int warpID = threadIdx.x / 32;
		const int lineID = threadIdx.x % 32;

      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag[8][8];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag[8][2];
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[8][2];

int batch_size =  32;
// for (int i = 0; i < Cout16; ++i) {
        // for (int i = 0; i < 8; ++i) {
				// 	for (int j = 0; j < 2; ++j) {
				// 		for (int t = 0; t < c_frag[0][0].num_elements; t++)
				// 		{

        //       c_frag[i][j].x[t] = (half)0;

				// 		}
				// 	}
				// }
		//for (int cin = 0; cin < Cin16; ++cin)  //?????这里应该是Cout16吧, c没有给值前不要load，会有乱数
        // for (int cout = 0; cout < 8; ++cout)
				// {
				// 	wmma::load_matrix_sync(c_frag[cout][0], c + 16 * cout, 0, wmma::mem_col_major);
				// 	wmma::load_matrix_sync(c_frag[cout][1], c + 16 * cout + 16 * 4*16, 0, wmma::mem_col_major);
				// }

        // for (int i = 0; i < Cin16; ++i) {
				// 	for (int j = 0; j < 2; ++j) {
				// 		for (int t = 0; t < b_frag[0][0].num_elements; t++)
				// 		{

        //       b_frag[i][j].x[t] = (half)0;


				// 		}
				// 	}
				// }
        // for (int i = 0; i < Cout16; ++i) {
				// 	for (int j = 0; j < Cin16; ++j) {
				// 		for (int t = 0; t < a_frag[0][0].num_elements; t++)
				// 		{

        //       // c_frag[i][j].x[t] = (half)0;
        //       // b_frag[i][j].x[t] = (half)0;
        //       a_frag[i][j].x[t] = (half)0;

				// 		}
				// 	}
				// }
/////////////////////////////////////
/// START - HIDDEN LAYERS
/////////////////////////////////////
for(int loop = 0; loop * gridDim.x*blockDim.x/32*batch_size < batchsizeTotal; ++loop){//  loop around whole chip
//int tmpBatchSize = batchsizeTotal - (loop+1) * gridDim.x*blockDim.x/32*batch_size;
int warpNum = blockDim.x/32;
if(loop * gridDim.x*warpNum*batch_size + blockIdx.x*warpNum*batch_size< batchsizeTotal){


int Cin16 = hiddenChannels[0];
int Cout16 = hiddenChannels[1];  
// if(threadIdx.x==0){
// printf("\n Cin %d\n", Cin16);}
// while(blockDim.x * i_d < Cin16*16*batch_size){
//     int idx = blockDim.x * i_d + threadIdx.x;
//     if(idx < Cin16*16*batch_size){
//         d[idx] = __float2half(input[0][idx]);
//     }
//     i_d += 1;
// }

//while(blockDim.x * i_d < Cin16*16*batch_size){
for(int i_d = 0; blockDim.x * i_d < Cin16*16*batch_size * warpNum; ++i_d){
    int idx = blockDim.x * i_d + threadIdx.x;
//     if(threadIdx.x==0){
// printf("\n Cin %d\n", blockDim.x);}
    if(idx < Cin16*16*batch_size * warpNum){
        d[idx] = __float2half(input[0][idx + loop * gridDim.x*Cin16*16*batch_size * warpNum + blockIdx.x*warpNum*Cin16*16*batch_size]);
        //printf(" %d", idx + loop * gridDim.x*Cin16*16*batch_size * warpNum + blockIdx.x*warpNum*Cin16*16*batch_size);
        
    }
}   

        //load B (input)
				for (int cin = 0; cin < Cin16; ++cin)
				{
					wmma::load_matrix_sync(b_frag[cin][0], warpID*Cin16*16*batch_size + d + 16 * cin, Cin16*16);
					wmma::load_matrix_sync(b_frag[cin][1], warpID*Cin16*16*batch_size + d + 16 * cin + 16 * Cin16*16, Cin16*16);
				}

/////////////////////////////////////
/// NETWORK - HIDDEN LAYERS START
/////////////////////////////////////
for (int hidden = 0; hidden <sizeof(hiddenChannels)/sizeof(hiddenChannels[0]) -1; ++hidden){
int currentWeightIndex = getCurrentWeightIndex(hidden);
int currentBiasIndex = getCurrentBiasIndex(hidden);
Cin16 = hiddenChannels[hidden];
Cout16 = hiddenChannels[hidden+1];

  //load A (weights)
int i_a = 0;
while(blockDim.x * i_a < Cin16*16*Cout16*16){
  
    int idx = blockDim.x * i_a + threadIdx.x;
    if(idx < Cin16*16*Cout16*16){
      // printf("get in %d",  idx);
        a[idx] = __float2half(weights[0][idx+currentWeightIndex]);//
    }
    i_a += 1;
}
 //load A (weights)
				for (int cout = 0; cout < Cout16; ++cout){
					for (int cin = 0; cin < Cin16; ++cin){
						wmma::load_matrix_sync(a_frag[cout][cin],
							a + 16 * cin + Cin16*16 * 16 * cout,
							Cin16*16);
          }
        }
  //load C to memory c(biass)
int i_c = 0;
while(blockDim.x * i_c < Cout16*16){
  
    int idx = blockDim.x * i_c + threadIdx.x;
    if(idx < Cout16*16){
      // printf("get in %d",  idx);
        c[idx] = __float2half(bias[0][idx+currentBiasIndex]);//
    }
    i_c += 1;
}
 //load C to fragment c_frag(bias)
    for (int cout = 0; cout < Cout16; ++cout)
  {
    wmma::load_matrix_sync(c_frag[cout][0], c + 16 * cout, 0, wmma::mem_col_major);
    wmma::load_matrix_sync(c_frag[cout][1], c + 16 * cout, 0, wmma::mem_col_major);
  }

if (hidden>0){
  //         //load B (input)
//#pragma unroll
				for (int cin = 0; cin < Cin16; ++cin)
				{
					wmma::load_matrix_sync(b_frag[cin][0], warpID*Cin16*16*batch_size + d + 16 * cin, Cin16*16);
					wmma::load_matrix_sync(b_frag[cin][1], warpID*Cin16*16*batch_size + d + 16 * cin + 16 * Cin16*16, Cin16*16);
				}
// //set b_frag to 0
	// 			for (int i = 0; i < Cin16; ++i) {
	// 				for (int j = 0; j < 2; ++j) {
	// 					for (int t = 0; t < b_frag[0][0].num_elements; t++)
	// 					{
  //             //half tmp = (half)1;//c_frag[i][j].x[t];
  //             // tmp = tmp + (half)5;
	// 						// c_frag[i][j].x[t] = tmp;
  //             b_frag[i][j].x[t] = (half)0;//c_frag[i][j].x[t];
              
              
	// 					}
	// 				}
	// 			}
  //       //load b_frag
  //       half* intermediateResultsforbfrag = d + 32 * Cin16*16 * warpID;
  // for (int cin = 0; cin < Cin16; ++cin)
	// 			{
	// 				wmma::load_matrix_sync(b_frag[cin][0], intermediateResultsforbfrag + 16 * cin, Cin16*16);
	// 				wmma::load_matrix_sync(b_frag[cin][1], intermediateResultsforbfrag + 16 * cin + 16 * Cin16*16, Cin16*16);
	// 			}

  //copy c_frag to b_frag, part 1
				// for (int i = 0; i < Cout16; ++i) {
				// 	for (int j = 0; j < 2; ++j) {
				// 		for (int t = 0; t < 4; t++)
				// 		{
        //       half tmp1 = c_frag[i][j].x[2*t];
        //       half tmp2 = c_frag[i][j].x[2*t+1];
        //       if (lineID%8 < 4){
        //         b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       }
        //       else{
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       }
				// 		}
				// 	}
				// }
        // for (int i = 0; i < Cout16; ++i){
        //   for (int t = 0; t < 4; t++){
				// 	for (int j = 0; j < 2; ++j)
				// 		//for (int t = 0; t < 4; t++)
				// 		{
        //       half tmp1 = c_frag[i][j].x[2*t];
        //       half tmp2 = c_frag[i][j].x[2*t+1];
        //       if (lineID%8 < 4){
        //         b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       }
        //       else{
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       }
        //       if (lineID%8 < 4){
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       }
        //       else{
        //         b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       }
				// 		}
				// 	}
				// }

        // //bellow are the new part
        // if (lineID%8 < 4){
        // for (int i = 0; i < Cout16; ++i){
        //   for (int t = 0; t < 4; t++){
				// 	for (int j = 0; j < 2; ++j)
				// 		//for (int t = 0; t < 4; t++)
				// 		{
        //       half tmp1 = c_frag[i][j].x[2*t];
        //       half tmp2 = c_frag[i][j].x[2*t+1];
        //       //if (lineID%8 < 4){
        //         b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       //}
        //       //else{
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       //}
        //       //if (lineID%8 < 4){
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       //}
        //       //else{
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       //}
				// 		}
				// 	}
        //   }}
        //   if (lineID%8 > 3){
        // for (int i = 0; i < Cout16; ++i){
        //   for (int t = 0; t < 4; t++){
				// 	for (int j = 0; j < 2; ++j)
				// 		//for (int t = 0; t < 4; t++)
				// 		{
        //       half tmp1 = c_frag[i][j].x[2*t];
        //       half tmp2 = c_frag[i][j].x[2*t+1];
        //       //if (lineID%8 < 4){
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       //}
        //       //else{
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       //}
        //       //if (lineID%8 < 4){
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       //}
        //       //else{
        //         b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       //}
				// 		}
				// 	}
        //   }}


        // //copy c_frag to b_frag, part 2
				// for (int i = 0; i < Cout16; ++i) {
				// 	for (int j = 0; j < 2; ++j) {
				// 		for (int t = 0; t < 4; t++)
				// 		{
        //       half tmp1 = c_frag[i][j].x[2*t];
        //       half tmp2 = c_frag[i][j].x[2*t+1];
        //       if (lineID%8 < 4){
        //         //b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) + 4 + lineID/8);
        //       }
        //       else{
        //         b_frag[i][j].x[2*t] = __shfl_sync(0xffffffff, tmp1, warpID*32 + 8*(lineID%4) + lineID/8);
        //         //b_frag[i][j].x[2*t+1] = __shfl_sync(0xffffffff, tmp2, warpID*32 + 8*(lineID%4) +4 + lineID/8);
        //       }
				// 		}
				// 	}
				// }
        // //copy part 3
        // // for (int i = 0; i < Cin16; ++i) {
				// // 	for (int j = 0; j < 2; ++j) {
				// // 		for (int t = 0; t < 8; t++)
				// // 		{
        // //       //half tmp = (half)1;//c_frag[i][j].x[t];
        // //       // tmp = tmp + (half)5;
				// // 			// c_frag[i][j].x[t] = tmp;
        // //       b_frag[i][j].x[t+8] = b_frag[i][j].x[t];
        // //       //printf("c.x[%d] is %.3f",t, __half2float(tmp));
              
				// // 		}
				// // 	}
				// // }

}

       //load C (bias)

      // // for (int i = 0; i < Cout16; ++i) {
      //   for (int i = 0; i < 8; ++i) {
			// 		for (int j = 0; j < 2; ++j) {
			// 			for (int t = 0; t < c_frag[0][0].num_elements; t++)
			// 			{

      //         c_frag[i][j].x[t] = (half)0;

			// 			}
			// 		}
			// 	}
// //#pragma unroll
// 				// for (int cin = 0; cin < Cin16; ++cin)  //?????这里应该是Cout16吧, c没有给值前不要load，会有乱数
//         for (int cout = 0; cout < Cout16; ++cout)
// 				{
// 					wmma::load_matrix_sync(c_frag[cout][0], c + 16 * cout, Cout16*16, wmma::mem_col_major);
// 					wmma::load_matrix_sync(c_frag[cout][1], c + 16 * cout + 16 * Cout16*16, Cout16*16, wmma::mem_col_major);
// 				}
//matmul
				for (int i = 0; i < Cout16; ++i) {
          //#pragma unroll
					for (int j = 0; j < 2; ++j) {
           //#pragma unroll
						for (int k = 0; k < Cin16; ++k) {
							wmma::mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
              //__syncthreads();
						}
					}
				}

        //activations
				for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < 2; ++j) {
						for (int t = 0; t < c_frag[0][0].num_elements; t++)
						{
              
              c_frag[i][j].x[t] = arbiacti1(c_frag[i][j].x[t]);
              
						}
					}
				}



        
        


        // //copy to shared new_frag
        // //half* intermediateResults = d + 32 * Cin16*16 * warpID;
        // //if (warpID==2){
        half* intermediateResults = d + batch_size * Cout16*16 * warpID;
        // //#pragma unroll
				for (int cout = 0; cout < Cout16; ++cout)
				{
					wmma::store_matrix_sync(intermediateResults + 16 * cout, c_frag[cout][0], Cout16*16, wmma::mem_col_major);
					wmma::store_matrix_sync(intermediateResults + 16 * cout + 16 * Cout16*16, c_frag[cout][1], Cout16*16, wmma::mem_col_major);
				}
        // //}//end if warpid = 1
//
//__syncwarp();
//printf("cout %d loop %d size of hidden cha%d\n", Cout16, loop, sizeof(hiddenChannels)/sizeof(hiddenChannels[0]) -1);
//printf("cout %d loop %d thread %d\n", Cout16, loop, threadIdx.x);
} //end for hidden layers
__syncthreads();
//__syncwarp();


for(int i_output = 0; blockDim.x * i_output < batch_size*Cout16*16 * warpNum; ++i_output){//
    int idx = blockDim.x * i_output + threadIdx.x;
    //if(idx < batch_size*Cout16*16 * warpNum){
       //output[0][idx] = __half2float(d[idx]);
       //output[0][idx] = __half2float((half(8.0)));
       //printf("thread %d < %d \n", idx, batch_size*Cout16*16 * warpNum);
       //printf("idx %d", idx + loop * gridDim.x*Cout16*16*batch_size * warpNum + blockIdx.x*warpNum*Cout16*16*batch_size);
        output[0][idx + loop * gridDim.x*Cout16*16*batch_size * warpNum + blockIdx.x*warpNum*Cout16*16*batch_size] = __half2float(d[idx]);
        
    //}
}
}//end of if
} //end of loop grid
// printf("thread %d", threadIdx.x);

if (warpID==0 && threadIdx.x == 0){
      for (int i = 0; i < output.size(0); i++) {
        for (int j = 0; j < output.size(1); j++) {
          //output[i][j] = __half2float(d[i+j*output.size(0)]);
          //output[0][j+i*output.size(0)] = __half2float(d[j+i*output.size(0)]);
        }
      }

      for (int i = 0; i < batch_size*8*16; ++i){
        //output[0][i] = __half2float(d[i]);
        // sWeightsHidden[i] = __(d[i]);
        // if (i % 32  == 0){ printf("\n");}
        // if (i % 32  == 0){ printf("\n");}
        //printf("d[%d], %.3f ", i, __half2float(d[i]));
        //printf("a[%d], %.3f ", i, __half2float(a[i]));

        //printf("weight[%d], %f", i, weights[0][i]);
      //   int out = (int)hiddenStructure[0];
      //   printf("test1[3], %d", test1[0]);
      // printf("a[%d], %.3f ", i, __half2float(b[i]));
      
      
        
      }
}
__syncthreads();


    if (warpID==0 && blockIdx.x ==0){
  printf("\n");
  //printf("texture value is %f", tex2D<float>(texObj, 3, 0));
  // half tmphalf = b1[3];
  // printf("a1 value is %f", __half2float(tmphalf));
  //printf("a1 value is %f", d1[0]);
for (int p = 0; p < 32; p++)
						{
  if (threadIdx.x == 32*warpID+p){
      
      // printf("fragment b is %.3f", __half2float(b_frag[7][0].x[3]));
      // printf("a[4000] is %.3f", __half2float(a[300]));
      // printf("fragment lenght is %d", a_frag[1][0].num_elements);
      //printf("thread [%d]" ,threadIdx.x);
      //test_frag[0][0].x[1] = b_frag[0][0].x[3];
      for (int t = 0; t < 8; t++)
						{
              
              //if (test_frag[0][0].x[t] == (half)0){
                //if (t < 8){
							//printf("thread[%d] frag_c.x[%d] is %.3f \n",p,t, __half2float(c_frag[0][0].x[t]));
              //printf("thread[%d] frag_b.x[%d] is %.3f \n",p,t, __half2float(test_frag[8][7].x[t]));
						 }
            //}
            printf("\n");
  }
            }

  }
  
    
}




    

