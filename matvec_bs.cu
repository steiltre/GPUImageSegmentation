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
#include <cuda_runtime.h>
#include <sys/time.h>

#include <cudpp.h>

//#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void expand_vector(float *d_NNZ_values, float *d_vec, unsigned *d_indices, float* d_expanded_vec,int NNZ, int dim){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NNZ){
        d_expanded_vec[i] = d_vec[d_indices[i]];
        //d_expanded_vec[i] = d_vec[i%2];
        //d_expanded_vec[i] = d_NNZ_values[i]*d_expanded_vec[i];
    }
}

__global__ void extract_vector(float *d_expanded_vec, float *d_vec, unsigned *d_rindices, int dim){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < dim){
        d_vec[i] = d_expanded_vec[d_rindices[i]-1];
        //d_expanded_vec[i] = d_vec[i%2];
        //d_expanded_vec[i] = d_NNZ_values[i]*d_expanded_vec[i];
    }
}
__global__ void vector_mul(float *d_NNZ_values, float *d_expanded_vec, float* d_vec,int NNZ, int dim){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NNZ){
        d_expanded_vec[i] = d_NNZ_values[i]*d_expanded_vec[i];
    }
    /*
    __syncthreads();
    if (i < dim){
        d_vec[i] = d_expanded_vec[2*i] + d_expanded_vec[2*i + 1];
    }
    */
}


/**
 * Host main routine
 */
int
main(int argc, char* argv[])
{
    int dim = atoi(argv[1]),NNZ = dim*2;
    // Allocate the host Arrays
    float *h_NNZ_values = (float *)malloc(sizeof(unsigned)*NNZ);
    unsigned *h_indices = (unsigned *)malloc(sizeof(unsigned)*NNZ);
    unsigned *h_rindices = (unsigned *)malloc(sizeof(unsigned)*dim);
    unsigned *h_flags = (unsigned *)malloc(sizeof(unsigned)*NNZ);
    float *h_vec = (float *)malloc(sizeof(unsigned)*dim);
    float *h_expanded_vec = (float *)malloc(sizeof(unsigned)*NNZ);
    float *h_scanned_vec = (float *)malloc(sizeof(unsigned)*NNZ);

    cudaEvent_t start, stop;
    float elapsedTime;
    // Allocate the host output vector C

    // Initialize the host input vectors
    for (int i = 0; i < dim; ++i){
        h_vec[i] = rand()/(float)RAND_MAX;
        h_rindices[i] = 2*(i+1);
    }
    for (int i = 0; i < NNZ; ++i){
        h_NNZ_values[i] = rand()/(float)RAND_MAX;
        if(i%2 == 0){
            h_indices[i] = 0;
            h_flags[i] = 1;
        } 
        if(i%2 == 1){
            h_indices[i] = 1;
            h_flags[i] = 0;
        }
    }
    printf("Done setting up Host arrays\n");
    // Allocate the device arrays
    float *d_NNZ_values,*d_vec, *d_expanded_vec,*d_scanned_vec;
    unsigned *d_indices,*d_flags,*d_rindices;
    
    cudaMalloc((void **)&d_NNZ_values, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_vec, sizeof(float)*dim);
    cudaMalloc((void **)&d_indices, sizeof(unsigned)*NNZ);
    cudaMalloc((void **)&d_rindices, sizeof(unsigned)*dim);
    cudaMalloc((void **)&d_flags, sizeof(unsigned)*NNZ);
    cudaMalloc((void **)&d_expanded_vec, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_scanned_vec, sizeof(float)*NNZ);
    //printf("Done allocating up device arrays\n");
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    cudaMemcpy(d_NNZ_values, h_NNZ_values, sizeof(float)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, sizeof(unsigned)*dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, sizeof(unsigned)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, sizeof(unsigned)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rindices, h_rindices, sizeof(unsigned)*dim, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(NNZ + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SEGMENTED_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, NNZ, 1, 0);  


    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    expand_vector<<<blocksPerGrid, threadsPerBlock>>>(d_NNZ_values, d_vec, d_indices, d_expanded_vec, NNZ, dim);
    vector_mul<<<blocksPerGrid, threadsPerBlock>>>(d_NNZ_values, d_expanded_vec, d_vec, NNZ, dim);
    cudppSegmentedScan(scanplan,d_scanned_vec,d_expanded_vec,d_flags,NNZ);
    extract_vector<<<blocksPerGrid, threadsPerBlock>>>(d_scanned_vec, d_vec, d_rindices, dim);
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //gettimeofday(&e,NULL);
    //timersub(&e, &s, &time_diff);
    //printf("Time to expand = %lf\n",time_diff.tv_usec*(1.0/1000000)+time_diff.tv_sec);
    cudaMemcpy(h_scanned_vec, d_scanned_vec, NNZ*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_expanded_vec, d_expanded_vec, NNZ*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Elapsed time : %f s\n" ,elapsedTime*0.001);
    for (int i = 0; i < dim; ++i){
        printf("%d --- %f\n",h_rindices[i],h_vec[i]);
    }
    printf("---------------------\n");
    for (int i = 0; i < NNZ; ++i){
        printf("%d %f - %f :: %f --- %d\n",h_indices[i],h_NNZ_values[i],h_expanded_vec[i],h_scanned_vec[i], h_flags[i]);
    }
    res = cudppDestroyPlan(scanplan);
    cudppDestroy(theCudpp);
    // Free device global memory
    cudaFree(d_NNZ_values);
    cudaFree(d_indices);
    cudaFree(d_rindices);
    cudaFree(d_vec);
    cudaFree(d_flags);
    cudaFree(d_expanded_vec);
    cudaFree(d_scanned_vec);

    // Free host memory
    free(h_NNZ_values);
    free(h_indices);
    free(h_rindices);
    free(h_vec);
    free(h_flags);
    free(h_expanded_vec);
    free(h_scanned_vec);

    printf("Done\n");
    return 0;
}

