
__shared__ half shmem[N];
__device__ half glmem[N];
//copy from global memory to shred memory
//suppose there're 32*k threads in a block, blockDim.x = 32*k
int i = 0;
while(blockDim.x * i * 2 < N){
    int idx = blockDim.x * i * 2 + threadIdx.x * 2;
    if(idx < N){
        shmem[idx] = glmem[idx];
    }
    if(idx + 1 < N){
        shmem[idx + 1] = glmem[idx + 1];
    }
    i += 1;
}

__shared__ float shmem[N];
__device__ float glmem[N];
//copy from global memory to shared memory
//suppose there're 32*k threads in a block, blockDim.x = 32*k
int i = 0;
while(blockDim.x * i < N){
    int idx = blockDim.x * i + threadIdx.x;
    if(idx < N){
        shmem[idx] = glmem[idx];
    }
    i += 1;
}