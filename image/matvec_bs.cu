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
#include <cudpp.h>
#include <cuda_runtime.h>
#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <sys/time.h>
extern "C" {
    #include "csr_mat.h"
}

__global__ void expand_vector(float *d_NNZ_values, float *d_vec, unsigned *d_indices, float* d_expanded_vec,int NNZ, int dim){
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
}

__global__ void extract_vector(float *d_expanded_vec, float *d_vec, unsigned *d_rindices, int dim){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < dim){
        d_vec[i] = d_expanded_vec[d_rindices[i]];
        i += blockDim.x*gridDim.x;
    }
}
__global__ void vector_mul(float *d_NNZ_values, float *d_expanded_vec, float* d_vec,int NNZ, int dim){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    while(i < NNZ){
        d_expanded_vec[i] = d_NNZ_values[i]*d_expanded_vec[i];
        i += blockDim.x*gridDim.x;
    }
}

/* 
    Taken from Nvidia cuda examples
*/
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i]*g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize]*g_idata[i+blockSize];

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
    if (tid == 0) g_odata[blockIdx.x] = sqrt(mySum/2);
    __syncthreads();
    tid = blockDim.x * blockIdx.x + threadIdx.x;
    float temp_val = g_odata[0];
    while(tid < n){
        g_idata[tid] /= temp_val;
        tid += blockDim.x*gridDim.x;
    }
} 

void matvec(const CUDPPHandle scanplan, float *d_NNZ_values, float *d_vec, unsigned *d_indices, unsigned *d_rindices, unsigned *d_flags, float* d_scanned_vec, float* d_expanded_vec,int NNZ, int dim){
    
    int threadsPerBlock = 256;
    //int blocksPerGrid =(NNZ + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = 4096;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    expand_vector<<<blocksPerGrid, threadsPerBlock>>>(d_NNZ_values, d_vec, d_indices, d_expanded_vec, NNZ, dim);
    
    //vector_mul<<<blocksPerGrid, threadsPerBlock>>>(d_NNZ_values, d_expanded_vec, d_vec, NNZ, dim);
    
    cudppSegmentedScan(scanplan,d_scanned_vec,d_expanded_vec,d_flags,NNZ);
    
    extract_vector<<<blocksPerGrid, threadsPerBlock>>>(d_scanned_vec, d_vec, d_rindices, dim);
}


void eigenvalue_solver(csr_mat *h_matrix, float *h_vec){
 
    cudaEvent_t start, stop;
    float elapsedTime;
    int NNZ = h_matrix->nnz;
    int dim = h_matrix->rows;
    float *h_expanded_vec = (float *)malloc(sizeof(unsigned)*NNZ);
    float *h_scanned_vec = (float *)malloc(sizeof(unsigned)*NNZ);
    float *d_NNZ_values,*d_vec, *d_expanded_vec,*d_scanned_vec, *d_norm, *d_temp_vec;
    unsigned *d_indices,*d_flags,*d_rindices;
    
    /*Copy stuff from host matrix to device */
    cudaMalloc((void **)&d_NNZ_values, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_indices, sizeof(unsigned)*NNZ);
    cudaMalloc((void **)&d_rindices, sizeof(unsigned)*(dim+1));
    cudaMalloc((void **)&d_flags, sizeof(unsigned)*NNZ);
    cudaMemcpy(d_NNZ_values, h_matrix->vals, sizeof(float)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_matrix->cols, sizeof(unsigned)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rindices, h_matrix->ptr, sizeof(unsigned)*(dim+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_matrix->flags, sizeof(unsigned)*NNZ, cudaMemcpyHostToDevice);

    /* Stuff for matvec operation */
    cudaMalloc((void **)&d_vec, sizeof(float)*dim);
    cudaMalloc((void **)&d_temp_vec, sizeof(float)*dim);
    cudaMalloc((void **)&d_expanded_vec, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_scanned_vec, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_norm, sizeof(float));

    cudaMemcpy(d_vec, h_vec, sizeof(unsigned)*dim, cudaMemcpyHostToDevice);
    // Launch the Vector Add CUDA Kernel

    printf("--------Initial Vector----------\n");  
    for (int i = 0; i < dim; ++i){
        printf("%f\n",h_vec[i]);
    }
    printf("---------------------\n");  
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SEGMENTED_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

    CUDPPHandle scanplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, NNZ, 1, 0);

    float *h_temp_vec = (float *)malloc(sizeof(unsigned)*dim);
    int threads = 256;
    int blocks = 4096;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    /*
        Reduction Usage
    */
    for(int count = 0; count < 20; count++){
        matvec(scanplan,d_NNZ_values, d_vec, d_indices, d_rindices, d_flags, d_scanned_vec,d_expanded_vec,NNZ,dim);
        cudaMemcpy(h_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);
        /*
        printf("Vector after matvec ---------------------\n");  
        for (int i = 0; i < dim; ++i){
            printf("%f\n",h_vec[i]);
        }
        printf("---------------------\n"); 
        */
        reduce6<float, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_vec, d_temp_vec, dim);
        cudaMemcpy(h_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);
        printf("Vector after normalizing---------------------\n");  
        for (int i = 0; i < dim; ++i){
            printf("%f\n",h_vec[i]);
        }
        printf("---------------------\n");  
    }
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);
    printf("Looks to be %f GFlops\n",((3*NNZ)*3*0.000000001)/(elapsedTime*0.001));
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    cudaMemcpy(h_scanned_vec, d_scanned_vec, NNZ*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_expanded_vec, d_expanded_vec, NNZ*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_vec, h_vec, sizeof(unsigned)*dim, cudaMemcpyHostToDevice);

    /*
    printf("h_vec after matvec\n");
    for (int i = 0; i < dim; ++i){
        printf("%f\n",h_vec[i]);
    }
    */

/*
    cudaMemcpy(h_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dim; ++i){
        printf("%d --- %f --- %f\n",h_matrix->ptr[i],h_vec[i],h_temp_vec[i]);
    }
    printf("---------------------\n");
*/ 
    /*
    for (int i = 0; i < NNZ; ++i){
        printf("%d %f - %f :: %f --- %d\n",h_matrix->cols[i],h_matrix->vals[i],h_expanded_vec[i],h_scanned_vec[i], h_matrix->flags[i]);
    }
    */
    res = cudppDestroyPlan(scanplan);
    cudppDestroy(theCudpp);
    // Free device global memory
    cudaFree(d_NNZ_values);
    cudaFree(d_indices);
    cudaFree(d_rindices);
    cudaFree(d_vec);
    cudaFree(d_temp_vec);
    cudaFree(d_flags);
    cudaFree(d_norm);
    cudaFree(d_expanded_vec);
    cudaFree(d_scanned_vec);  

    free(h_expanded_vec);
    free(h_temp_vec);
    free(h_scanned_vec);  
}

/**
 * Host main routine
 */
int main(int argc, char* argv[])
{
    int dim = atoi(argv[1]),NNZ = dim;
    // Allocate the host Arrays

    csr_mat *h_matrix = csr_alloc(NNZ,dim);
    /*
    float *h_NNZ_values = (float *)malloc(sizeof(unsigned)*NNZ);
    unsigned *h_indices = (unsigned *)malloc(sizeof(unsigned)*NNZ);
    unsigned *h_rindices = (unsigned *)malloc(sizeof(unsigned)*dim);
    unsigned *h_flags = (unsigned *)malloc(sizeof(unsigned)*NNZ);
    */
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

    for (int i = 0; i < NNZ; ++i){
        printf("%d :: %d :: %f :: %d\n",h_matrix->ptr[i],h_matrix->cols[i],h_matrix->vals[i],h_matrix->flags[i]);
    }

    printf("Done setting up Host arrays\n");
    // Allocate the device arrays
    eigenvalue_solver(h_matrix,h_vec);
    // Free host memory
    csr_free(h_matrix);

    free(h_vec);

    return 0;
}

