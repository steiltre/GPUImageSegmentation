
#include <stdlib.h>
#include <stdio.h>


extern "C"{
    #include "power_iteration.h"
}
#define NUM_THREAD 256
#define NUM_BLOCK 256
__global__ void
reduce(float *g_idata, float *g_odata, unsigned int n){

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i == 0){
        g_odata[0] = 0;
    }
    __syncthreads();
    while(i < n){
        atomicAdd(g_odata, g_idata[i]);
        i += blockDim.x*gridDim.x;
    }
}

__global__ void scalar_prod(float *g_idata_a, float *g_idata_b, float *g_odata, unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i == 0){
        g_odata[0] = 0;
    }
    __syncthreads();
    while(i < n){
        atomicAdd(g_odata, g_idata_a[i]*g_idata_b[i]);
        i += blockDim.x*gridDim.x;
    }
}


/*
  Modification of the reduce3 code provided in the Cuda examples
  Computes the norm, and normalizes the vector
  g_idata is the input vector
  g_odata stores the value of the norm
  n = size of row
*/

__global__ void normalize_vector(float *g_idata, float *g_odata, unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i == 0){
        g_odata[0] = 0;
    }
    __syncthreads();
    while(i < n){
        atomicAdd(g_odata, g_idata[i]*g_idata[i]);
        i += blockDim.x*gridDim.x;
    }
    i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i == 0){
        g_odata[0] = sqrt(g_odata[0]);
    }
    cudaDeviceSynchronize();
    i = blockDim.x * blockIdx.x + threadIdx.x;
    float temp = g_odata[0];
    while(i < n){
       g_idata[i] /= temp;
        i += blockDim.x*gridDim.x;
    }

}

__global__ void compute_evector(float *g_idata, unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < n){
        g_idata[i] = sqrt(g_idata[i]);
        i += blockDim.x*gridDim.x;
    }
}

__global__ void mat_vec(float *d_NNZ_values, float *d_vec, unsigned *d_indices, float* d_expanded_vec, unsigned *d_scan_ind, float *d_projvec, float *d_norm, float *d_diag, float *d_evec, int NNZ, int dim){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i == 0){
        scalar_prod<<< NUM_BLOCK, NUM_THREAD>>>(d_vec, d_evec, d_norm, dim);
        //reduce<<< NUM_BLOCK, NUM_THREAD>>>(d_vec, d_norm, dim);
        //cudaDeviceSynchronize();
        //printf("scalar product = %f\n",d_norm[0]);
    }
    cudaDeviceSynchronize();
    while(i < NNZ){
        d_expanded_vec[i] = d_vec[d_indices[i]]/sqrt(d_diag[d_indices[i]]);
        i += blockDim.x*gridDim.x;
    }
    __syncthreads();

    i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < NNZ){
        d_expanded_vec[i] = d_NNZ_values[i]*d_expanded_vec[i];
        i += blockDim.x*gridDim.x;
    }
    __syncthreads();
    i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j,k,start;
    float mySum;

    while(i < dim){
        k = d_scan_ind[2*i+1];
        start = d_scan_ind[2*i];
        mySum = 0;
        for(j = 0; j < k; j++){
          mySum += d_expanded_vec[start+j];  
        }
        d_projvec[i] = d_vec[i];
        d_vec[i] = mySum/sqrt(d_diag[i]);
        i += blockDim.x*gridDim.x;
    }
    cudaDeviceSynchronize();

    
    i = blockDim.x * blockIdx.x + threadIdx.x;
    mySum = d_norm[0];
    while(i < dim){
        //printf("%d - %f\n",i,(2.0/dim)*(d_projvec[i]-mySum));
        d_vec[i] = 2*(d_projvec[i]-mySum*d_evec[i]) - d_vec[i];
        i += blockDim.x*gridDim.x;
    }
    cudaDeviceSynchronize();
    i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i == 0){
      normalize_vector<<< NUM_BLOCK, NUM_THREAD>>>(d_vec, d_norm, dim);
      /*
      scalar_prod<<< NUM_BLOCK, NUM_THREAD>>>(d_vec, d_projvec, d_norm, dim);
      printf("Eigenvalue seems to be: %f\n",2-d_norm[0]);
      */
    }
    
 
}

void eigenvalue_solver(csr_mat *h_matrix, float *h_vec, float *h_diag){
 
    cudaEvent_t start, stop;
    float elapsedTime;
    int NNZ = h_matrix->nnz;
    int dim = h_matrix->rows;

    int j = 0,count = 1;
    h_matrix->flags[2*j]= 0;
    for(int i = 1; i < NNZ; i++){
        if(h_matrix->cols[i] < h_matrix->cols[i-1]){
            h_matrix->flags[2*j+1] = count;
            count = 0;
            j++; 
            h_matrix->flags[2*j] = i;
        }
        count++;
    }
    h_matrix->flags[2*j+1] = count;

    float *d_NNZ_values,*d_vec, *d_evec, *d_expanded_vec,*d_scanned_vec, *d_projvec, *d_norm, *d_diag;
    unsigned *d_indices,*d_flags,*d_rindices,*d_scan_ind;
    
    /*Copy stuff from host matrix to device */
    cudaMalloc((void **)&d_NNZ_values, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_indices, sizeof(unsigned)*NNZ);
    cudaMalloc((void **)&d_rindices, sizeof(unsigned)*(dim+1));
    cudaMalloc((void **)&d_flags, sizeof(unsigned)*NNZ);
    cudaMalloc((void **)&d_scan_ind, sizeof(unsigned)*2*dim);
    cudaMemcpy(d_NNZ_values, h_matrix->vals, sizeof(float)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_matrix->cols, sizeof(unsigned)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rindices, h_matrix->ptr, sizeof(unsigned)*(dim+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_matrix->flags, sizeof(unsigned)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scan_ind, h_matrix->flags, sizeof(unsigned)*2*dim, cudaMemcpyHostToDevice);
    /* Stuff for matvec operation */
    cudaMalloc((void **)&d_vec, sizeof(float)*dim);
    cudaMalloc((void **)&d_evec, sizeof(float)*dim);
    cudaMemcpy(d_evec, h_diag, sizeof(unsigned)*dim, cudaMemcpyHostToDevice);  
    cudaMalloc((void **)&d_expanded_vec, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_scanned_vec, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_norm, sizeof(float));
    cudaMalloc((void **)&d_projvec, sizeof(float)*dim);
    cudaMemcpy(d_vec, h_vec, sizeof(unsigned)*dim, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_diag, sizeof(unsigned)*dim);
    cudaMemcpy(d_diag, h_diag, sizeof(unsigned)*dim, cudaMemcpyHostToDevice);    

    /*
    printf("--------Initial Vector----------\n");  
    for (int i = 0; i < dim; ++i){
        printf("%f\n",h_vec[i]);
    }
    printf("---------------------\n");  
    */



    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    compute_evector<<< NUM_BLOCK, NUM_THREAD>>>(d_evec, dim);
    normalize_vector<<< NUM_BLOCK, NUM_THREAD>>>(d_evec, d_norm, dim);
    for(int count = 0; count < 30; count++){
        //matvec(scanplan,d_NNZ_values, d_vec, d_indices, d_rindices, d_flags, d_scanned_vec,d_expanded_vec,NNZ,dim);   
        //scalar_prod<float><<< dimGrid, dimBlock, smemSize >>>(d_vec, d_vec, d_preduc, dim);
        //cudaDeviceSynchronize(); 
        //reduce<<< NUM_BLOCK, NUM_THREAD>>>(d_vec, d_preduc, dim);
        //cudaDeviceSynchronize(); 
        mat_vec<<<NUM_BLOCK, NUM_THREAD>>>(d_NNZ_values, d_vec, d_indices, d_expanded_vec,d_scan_ind, d_projvec, d_norm, d_diag, d_evec, NNZ, dim);

        /*
        cudaMemcpy(h_expanded_vec, d_preduc, NUM_BLOCK*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_scanned_vec, d_norm, sizeof(float), cudaMemcpyDeviceToHost);  
        printf("Partial Reduction results ---------------------\n");  
        //printf("Dot product is %f\n",h_scanned_vec[0]);
        for (int i = 0; i < NUM_BLOCK; ++i){
            printf("%f\n",h_expanded_vec[i]);
        }
        printf("---------------------\n");
        */
        // Used to see vector after each iteration
        /*
        cudaMemcpy(h_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);
        printf("Vector after iteration %d\n",count);  
        for (int i = 0; i < dim; ++i){
            printf("%f\n",h_vec[i]);
        }
        printf("---------------------\n");
        */
    }
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);
    printf("Looks to be %f GFlops\n",((2*NNZ)*3*0.000001)/(elapsedTime));
    cudaMemcpy(h_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.

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
    // Free device global memory
    cudaFree(d_NNZ_values);
    cudaFree(d_indices);
    cudaFree(d_rindices);
    cudaFree(d_vec);
    cudaFree(d_evec);
    cudaFree(d_flags);
    cudaFree(d_diag);
    cudaFree(d_norm);
    cudaFree(d_projvec);
    cudaFree(d_expanded_vec);
    cudaFree(d_scanned_vec);  
    cudaFree(d_scan_ind);
 
}
