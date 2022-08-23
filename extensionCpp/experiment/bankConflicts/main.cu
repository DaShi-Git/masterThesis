//make
//./main.exe
//nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict ./main.exe

//nvcc -g -G main.cu -o main
//nvprof ./main


#include<stdio.h>
#include<time.h>
#define WARPSIZE 32
__global__ void kernel1(float* A) {
    __shared__ float data[32][32];
    int tid = threadIdx.x;
    int col = tid/WARPSIZE;
    int row = tid%WARPSIZE;
    // int row = tid/WARPSIZE;
    // int col = tid%WARPSIZE;
    data[row][col] = 100.f;
    A[tid] = data[row][col];
}


__global__ void kernel2(float* A) {
    __shared__ float data[32][32];
    int tid = threadIdx.x;
    int row = tid/WARPSIZE;
    int col = tid%WARPSIZE;
    data[row][col] = 100.f;
    A[tid] = data[row][col];
}

__global__ void warmup(float* A) {
    __shared__ float data[32][32];
    int tid = threadIdx.x;
    int col = tid/WARPSIZE;
    int row = tid%WARPSIZE;
    data[row][col] = 100.f;
    A[tid] = data[row][col];
}
    
void checkValue(float* A, int len, int val = 100.f) {
    for(int i = 0; i < len; i++) {
        if(A[i] != val) {
            printf("Error accured");
        }
    }
}

int main() {
    clock_t start, end;
    int blocksize = 32*32;
    float* h_A = (float*)malloc(sizeof(float)*blocksize);
    float* d_A;
    cudaMalloc(&d_A, sizeof(float)*blocksize);
    start = clock();
    warmup<<<1, blocksize>>>(d_A);
    cudaDeviceSynchronize();
    end = clock();
    printf("warmup : %f\n",(double)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(h_A, d_A, blocksize*sizeof(float), cudaMemcpyDeviceToHost);
    checkValue(h_A, blocksize);
    
    
    
    start = clock();
    kernel2<<<1, blocksize>>>(d_A);
    cudaDeviceSynchronize();
    end = clock();
    printf("kernel2: %f\n",(double)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(h_A, d_A, blocksize*sizeof(float), cudaMemcpyDeviceToHost);    
    checkValue(h_A, blocksize);

    start = clock();
    kernel1<<<1, blocksize>>>(d_A);
    cudaDeviceSynchronize();
    end = clock();
    printf("kernel1: %f\n",(double)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(h_A, d_A, blocksize*sizeof(float), cudaMemcpyDeviceToHost);
    checkValue(h_A, blocksize);
    cudaFree(d_A);
    free(h_A);
    return 0;
}