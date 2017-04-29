/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */
#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <sys/time.h>
extern "C" {
    #include "./src/power_iteration.cuh"
}


/*
__global__ void power_iteration(float *d_NNZ_values, float *d_vec, unsigned *d_indices, unsigned *d_rindices, float* d_expanded_vec,float *d_norm,int NNZ, int dim, bool nIsPow2, int blockSize){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float d_expanded_vec_shared[256];
    __shared__ float d_NNZ_values_shared[256];

    while(i < NNZ){
        d_expanded_vec_shared[i%256] = d_vec[d_indices[i]];
        d_NNZ_values_shared[i%256] = d_NNZ_values[i];
        //d_expanded_vec[i] = d_vec[d_indices[i]];
        //d_NNZ_values[i] = d_NNZ_values[i];
        i += blockDim.x*gridDim.x;
    }
    __syncthreads();
    i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < NNZ){
        d_expanded_vec_shared[i%256] = d_NNZ_values_shared[i%256]*d_expanded_vec_shared[i%256];
        //d_expanded_vec[i] *= d_NNZ_values[i];
        i += blockDim.x*gridDim.x;
    }
    __syncthreads();
    i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < NNZ){
        d_expanded_vec[i] = d_expanded_vec_shared[i%256];
        i += blockDim.x*gridDim.x;
    }

    __syncthreads();
    i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < dim){
        
        if(i == 0){
            for(int j = 0; j <= d_rindices[1]; j++ ){
                d_vec[i] += d_expanded_vec[j]; 
            }
        }
        else{
            for(int j = d_rindices[i-1]; j <= d_rindices[i]; j++ ){
                d_vec[i] += d_expanded_vec[j]; 
            }
        }
        i += blockDim.x*gridDim.x;
    }
    __syncthreads();

    float *sdata = SharedMemory<float>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < dim)
    {
        mySum += d_vec[i]*d_vec[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < dim)
            mySum += d_vec[i+blockSize]*d_vec[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) d_norm[0] = sqrt(mySum/2);
    __syncthreads();
    tid = blockDim.x * blockIdx.x + threadIdx.x;
    float temp_val = d_norm[0];
    while(tid < dim){
        d_vec[tid] /= temp_val;
        tid += blockDim.x*gridDim.x;
    }

}
*/





/**
 * Host main routine
 */
int main(int argc, char* argv[])
{
    int dim = atoi(argv[1]),NNZ = dim;
    // Allocate the host Arrays

    csr_mat *h_matrix = csr_alloc(NNZ,dim);
    
    float *h_vec = (float *)malloc(sizeof(unsigned)*dim);

    // Allocate the host output vector C

    // Initialize the host input vectors
    for (int i = 0; i < dim; ++i){
        h_vec[i] = 1;
        //printf("%f\n",h_vec[i]);
        h_matrix->ptr[i] = i;
    }
    h_matrix->ptr[dim] = dim;
    for (int i = 0; i < dim; ++i){
        h_matrix->vals[i] = i+1;
        h_matrix->cols[i] = i;
        h_matrix->flags[i] = 1;
    }
/*
    for (int i = 0; i < NNZ; ++i){
        printf("%d :: %d :: %f :: %d\n",h_matrix->ptr[i],h_matrix->cols[i],h_matrix->vals[i],h_matrix->flags[i]);
    }
*/
    printf("Done setting up Host arrays\n");
    // Allocate the device arrays
    eigenvalue_solver(h_matrix,h_vec);
    // Free host memory
    csr_free(h_matrix);

    free(h_vec);

    return 0;
}

