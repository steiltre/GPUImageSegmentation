
#include <stdlib.h>
#include <stdio.h>


extern "C"{
    #include "power_iteration.h"
}
#define NUM_THREAD 256
#define NUM_BLOCK 256


__global__ void scalar_prod(float *g_idata_a, float *g_idata_b, float *g_odata, unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i == 0){
        //atomicAnd((int*)g_odata, 0);
    }
    float temp;
    while(i < n){
        temp = g_idata_a[i]*g_idata_b[i];
        atomicAdd(g_odata, temp);        
        i += blockDim.x*gridDim.x;
    }
}

__global__ void normalize_vector(float *g_idata, float *g_odata, unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i == 0){
        //atomicAnd((int*)g_odata, 0);
    }
    float temp;
    while(i < n){
        temp = g_idata[i]*g_idata[i];
        atomicAdd(g_odata, temp);
        i += blockDim.x*gridDim.x;
    }
}

__global__ void scale_vector(float *g_idata, float *g_odata, unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    float temp = g_odata[0];
    while(i < n){
        g_idata[i] = g_idata[i]/sqrt(temp);
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

__global__ void set_zero(float *g_idata){
    g_idata[0] = 0;
}

__global__ void sub_vector(float *d_vec, float *d_projvec, unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < n){
        d_vec[i] = d_projvec[i] - d_vec[i];
        i += blockDim.x*gridDim.x;
    }
}

__global__ void compute_projection(float *d_projvec,float *d_vec,float *d_evec,float *d_norm,unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    float mySum = d_norm[0];
    while(i < n){
        d_projvec[i] = 2*(d_vec[i]-mySum*d_evec[i]);
        i +=blockDim.x*gridDim.x;
    }
}

__global__ void seg_scan(float *d_vec,float *d_expanded_vec,float *d_diag, unsigned *d_scan_ind,unsigned int dim){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j,k,start;
    float mySum;
    while(i < dim){
        k = d_scan_ind[2*i+1];
        start = d_scan_ind[2*i];
        mySum = 0;
        for(j = 0; j < k; j++){
          mySum += d_expanded_vec[start+j];  
        }
        d_vec[i] = mySum/sqrt(d_diag[i]);
        i += blockDim.x*gridDim.x;
    }
}

__global__ void mat_vec(float *d_NNZ_values, float *d_vec, unsigned *d_indices, float* d_expanded_vec, unsigned *d_scan_ind, float *d_norm, float *d_diag, int NNZ, int dim){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float temp = 0;
    if(i == 0){
        printf("scalar product = %f\n",d_norm[0]);
    }
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
}

void eigenvalue_solver(csr_mat *h_matrix, float *h_vec, float *h_diag){
 
    cudaEvent_t start, stop;
    float elapsedTime;
    int NNZ = h_matrix->nnz;
    int dim = h_matrix->rows;
    float temp_sum = 0, temp_sum2= 0;
    for(int i = 0; i < dim; i++){
        h_matrix->flags[2*i] = h_matrix->ptr[i];
        h_matrix->flags[2*i+1] = h_matrix->ptr[i+1]-h_matrix->ptr[i];
        temp_sum += h_diag[i];
    }
    for(int i = 0; i < dim; i++){
        temp_sum2 += sqrt(h_diag[i]/temp_sum);
    }
    printf("Dot product should be %f\n",temp_sum2);
    float *d_NNZ_values,*d_vec, *d_evec, *d_expanded_vec, *d_projvec, *d_norm, *d_diag;
    unsigned *d_indices,*d_scan_ind;
    
    /*Copy stuff from host matrix to device */
    cudaMalloc((void **)&d_NNZ_values, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_indices, sizeof(unsigned)*NNZ);
    cudaMalloc((void **)&d_scan_ind, sizeof(unsigned)*2*dim);

    cudaMalloc((void **)&d_vec, sizeof(float)*dim);
    cudaMalloc((void **)&d_expanded_vec, sizeof(float)*NNZ);
    cudaMalloc((void **)&d_norm, sizeof(float));
    cudaMalloc((void **)&d_projvec, sizeof(float)*dim);
    cudaMalloc((void **)&d_diag, sizeof(float)*dim);
    cudaMalloc((void **)&d_evec, sizeof(float)*dim);


    cudaMemcpy(d_NNZ_values, h_matrix->vals, sizeof(float)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_matrix->cols, sizeof(unsigned)*NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scan_ind, h_matrix->flags, sizeof(unsigned)*2*dim, cudaMemcpyHostToDevice);
    /* Stuff for matvec operation */
    cudaMemcpy(d_evec, h_diag, sizeof(float)*dim, cudaMemcpyHostToDevice);  
    cudaMemcpy(d_vec, h_vec, sizeof(float)*dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_diag, h_diag, sizeof(float)*dim, cudaMemcpyHostToDevice);  

// --------------------------------------------------------------------------------
    float h_norm;
    float *temp_vec = (float*)malloc(sizeof(float)*dim);
    
    set_zero<<<1,1>>>(d_norm);
    compute_evector<<<NUM_BLOCK, NUM_THREAD>>>(d_evec, dim);
    normalize_vector<<< NUM_BLOCK, NUM_THREAD>>>(d_evec, d_norm, dim);
    scale_vector<<< NUM_BLOCK, NUM_THREAD>>>(d_evec, d_norm, dim);


    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    for(int count = 0; count < 200; count++){
        cudaMemcpy(temp_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);
        set_zero<<<1,1>>>(d_norm);
        scalar_prod<<<NUM_BLOCK, NUM_THREAD>>>(d_vec,d_evec,d_norm,dim);
        compute_projection<<<NUM_BLOCK, NUM_THREAD>>>(d_projvec,d_vec,d_evec,d_norm,dim);

        mat_vec<<<NUM_BLOCK, NUM_THREAD>>>(d_NNZ_values, d_vec, d_indices, d_expanded_vec,d_scan_ind, d_norm, d_diag, NNZ, dim);
        seg_scan<<<NUM_BLOCK, NUM_THREAD>>>(d_vec,d_expanded_vec,d_diag,d_scan_ind,dim);
        
        sub_vector<<<NUM_BLOCK, NUM_THREAD>>>(d_vec,d_projvec,dim);
        
        set_zero<<<1,1>>>(d_norm);
        normalize_vector<<< NUM_BLOCK, NUM_THREAD>>>(d_vec, d_norm, dim);
        scale_vector<<< NUM_BLOCK, NUM_THREAD>>>(d_vec, d_norm, dim);
        
        /*
        cudaMemcpy(d_projvec, temp_vec, dim*sizeof(float), cudaMemcpyHostToDevice);
        scalar_prod<<<NUM_BLOCK, NUM_THREAD>>>(d_vec,d_projvec,d_norm,dim);
        cudaMemcpy(&temp_sum, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Eigenvalue seems to be %f\n",2-temp_sum);
        */
        //normalize_vector<<< NUM_BLOCK, NUM_THREAD>>>(d_vec, d_norm, dim);
        // Used to see vector after each iteration
        
        cudaMemcpy(h_vec, d_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);
        printf("Vector after iteration %d\n",count);  
        for (int i = 0; i < 5; ++i){
            printf("%d :: %f\n",i,h_vec[i]);
        }
        printf("---------------------\n");
        
        
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
    //cudaFree(d_rindices);
    cudaFree(d_vec);
    cudaFree(d_evec);
    cudaFree(d_diag);
    cudaFree(d_norm);
    cudaFree(d_projvec);
    cudaFree(d_expanded_vec); 
    cudaFree(d_scan_ind);
 
}
