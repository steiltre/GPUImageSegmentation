#define THREADSPERBLOCK 256

#include <stdlib.h>
#include <stdio.h>

extern "C"
{
#include "diag_mat.h"
#include "matvec.h"
#include "image_segmentation.h"
}
#include "matvec.cuh"

__device__ float iter_diff; /* Norm of difference between successive iterates */
__device__ float d_norm;
__device__ float d_lambda_max;
__device__ float d_factor;
__device__ float d_vector_sum;
__device__ float d_proj_inner;

__global__ void vec_norm(
        float * vec,
        int n)
{

        if (threadIdx.x + blockDim.x*blockIdx.x < n) { //+ blockDim.x*gridDim.x*i < n) {
            atomicAdd(&d_norm, vec[ threadIdx.x + blockDim.x*blockIdx.x ] * vec[ threadIdx.x + blockDim.x*blockIdx.x ]);
        }
}


__global__ void normalize(
        float * vec,
        int n)
{
    // Kernel to normalize vector

        if (threadIdx.x + blockDim.x*blockIdx.x  < n) { //+ blockDim.x*gridDim.x*i < n) {
            vec[ threadIdx.x + blockDim.x*blockIdx.x ] = d_factor * vec[ threadIdx.x + blockDim.x*blockIdx.x ];
        }
}

__global__ void norm_diff(
        float * vec1,
        float * vec2,
        int n)
{
    // Kernel to compute norm of difference between vectors

        if (threadIdx.x + blockDim.x*blockIdx.x < n) { //+ blockDim.x*gridDim.x*i < n) {
            atomicAdd( &iter_diff, ( vec1[ threadIdx.x + blockDim.x*blockIdx.x ] - vec2[ threadIdx.x + blockDim.x*blockIdx.x ] )
                * ( vec1[ threadIdx.x + blockDim.x*blockIdx.x ] - vec2[ threadIdx.x + blockDim.x*blockIdx.x ] ) );
        }
}

__global__ void calculate_eigenvalue_inner_prod(
        float * Avec,
        float * vec,
        int n)
{
        if (threadIdx.x + blockDim.x*blockIdx.x < n) { //+ blockDim.x*gridDim.x*i < n) {
            atomicAdd(&d_lambda_max, Avec[ threadIdx.x + blockDim.x*blockIdx.x ] * vec[ threadIdx.x + blockDim.x*blockIdx.x ]);
        }


}

__global__ void calculate_eigenvalue()
{
        if (threadIdx.x + blockDim.x*blockIdx.x == 0)
        {
            d_lambda_max = d_lambda_max / d_norm;
        }
}


__global__ void sum_vector(
        float * vec,
        int n)
{
    if (threadIdx.x + blockDim.x*blockIdx.x < n) {
        atomicAdd(&d_vector_sum, vec[ threadIdx.x + blockDim.x*blockIdx.x ]);
    }
}

__global__ void add_shift(
        float * vec,
        float * vec_old,
        int n)
{
    if (threadIdx.x + blockDim.x*blockIdx.x < n) {
        vec[ threadIdx.x + blockDim.x*blockIdx.x ] = d_lambda_max * vec_old[threadIdx.x + blockDim.x*blockIdx.x] - vec[ threadIdx.x + blockDim.x*blockIdx.x ];
    }
}

__global__ void zero_proj()
{
    d_proj_inner = 0;
}

__global__ void proj_inner(
        float * vec,
        float * diag,
        int n)
{
    if (threadIdx.x + blockDim.x*blockIdx.x < n) {
        atomicAdd(&d_proj_inner, diag[threadIdx.x + blockDim.x*blockIdx.x] * vec[ threadIdx.x + blockDim.x*blockIdx.x ]);
    }
}

__global__ void add_projection(
        float * vec,
        float * vec_old,
        float * diag,
        int n)
{

    if (threadIdx.x + blockDim.x*blockIdx.x < n) {
        vec[ threadIdx.x + blockDim.x*blockIdx.x ] -= diag[ threadIdx.x + blockDim.x*blockIdx.x ] * d_proj_inner;
    }

}

__global__ void initialize_guess_vector(
        float * vec,
        int n)
{

        if (threadIdx.x + blockDim.x*blockIdx.x < n) { //+ blockDim.x*gridDim.x*i < n) {
            if ( (threadIdx.x + blockDim.x*blockIdx.x)%2 == 0) { //+ blockDim.x*gridDim.x*i == 0) {
                vec[ threadIdx.x + blockDim.x*blockIdx.x ] = sqrtf(2)/sqrtf(n);
            }
            else {
                vec[ threadIdx.x + blockDim.x*blockIdx.x ] = 0;
            }
        }

}

__global__ void zero_vector(
        float * vec,
        int n)
{
    if (threadIdx.x + blockDim.x*blockIdx.x < n) {
        vec[ threadIdx.x + blockDim.x*blockIdx.x ] = 0.0;
    }
}

__global__ void manip_norm()
{
    d_factor = 1.0 / sqrt(d_norm);
}

__global__ void zero_norm()
{
    d_norm = 0.0;
    iter_diff = 0.0;
}

__global__ void zero_eig()
{
    d_lambda_max = 0.0;
}

float * segmentation(
        diag_mat * mat)
{

    float * vecout;
    float lambda_max;
    float * d_vec1, * d_vec2;
    float * d_vectemp, * d_vectemp2;
    float * d_mat, * d_diag, *d_diag_inv;
    int * d_offset;
    int * d_no_offset;
    float tol = 10e-8;
    int max_iter = 1000;
    float diff;
    int i;
    int cont;

    diag_mat * main_diag = diag_sqrt(mat);  /* Diagonal entries used in generalized eigenvalue problem */
    diag_mat * diag_inv = diag_inv_sqrt(mat);

    vecout = (float*) malloc( mat->width * sizeof(*vecout) );

    cudaMalloc( (void **) &d_vec1, mat->width * sizeof(float) );
    cudaMalloc( (void **) &d_vec2, mat->width * sizeof(float) );
    cudaMalloc( (void **) &d_vectemp, mat->width * sizeof(float) );
    cudaMalloc( (void **) &d_vectemp2, mat->width * sizeof(float) );
    cudaMalloc( (void **) &d_mat, mat->nnz * sizeof(float) );
    cudaMalloc( (void **) &d_diag, main_diag->nnz * sizeof(float) );
    cudaMalloc( (void **) &d_diag_inv, main_diag->nnz * sizeof(float) );
    cudaMalloc( (void **) &d_offset, mat->ndiags * sizeof(int) );
    cudaMalloc( (void **) &d_no_offset, sizeof(int) );

    cudaMemcpy( d_mat, mat->vals, mat->nnz * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( d_diag, main_diag->vals, main_diag->nnz * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( d_diag_inv, diag_inv->vals, diag_inv->nnz * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( d_offset, mat->offset, mat->ndiags * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( d_no_offset, main_diag->offset, sizeof(int), cudaMemcpyHostToDevice );

    initialize_guess_vector<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, mat->width);
    zero_vector<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, mat->width);
    zero_vector<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vectemp, mat->width);
    zero_vector<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vectemp2, mat->width);

    i = 0;
    cont = 1;

    /* Calculate largest eigenvalue for shifting spectrum */
    while (cont == 1) {

        if (i%2 == 0) {
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_diag_inv, d_vec1, d_vectemp, d_no_offset, diag_inv->ndiags, diag_inv->width);
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_mat, d_vectemp, d_vectemp2, d_offset, mat->ndiags, mat->width);
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_diag_inv, d_vectemp2, d_vec2, d_no_offset, diag_inv->ndiags, diag_inv->width);
            zero_norm<<<1,1>>>();
            zero_eig<<<1,1>>>();
            calculate_eigenvalue_inner_prod<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>( d_vec1, d_vec2, mat->width );
            vec_norm<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, mat->width);
            manip_norm<<<1,1>>>();
            normalize<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, mat->width);
        }
        else {
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_diag_inv, d_vec2, d_vectemp, d_no_offset, diag_inv->ndiags, diag_inv->width);
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_mat, d_vectemp, d_vectemp2, d_offset, mat->ndiags, mat->width);
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_diag_inv, d_vectemp2, d_vec1, d_no_offset, diag_inv->ndiags, diag_inv->width);
            zero_norm<<<1,1>>>();
            zero_eig<<<1,1>>>();
            calculate_eigenvalue_inner_prod<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>( d_vec1, d_vec2, mat->width );
            vec_norm<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, mat->width);
            manip_norm<<<1,1>>>();
            normalize<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, mat->width);
        }

        norm_diff<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, d_vec2, mat->width);

        cudaMemcpyFromSymbol( &diff, iter_diff, sizeof(iter_diff) );
        cudaMemcpyFromSymbol( &lambda_max, d_lambda_max, sizeof(d_lambda_max) );

        i++;

        if ( diff < tol || i > max_iter ) {
            cont = 0;
        }
    }

    initialize_guess_vector<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, mat->width);
    zero_vector<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, mat->width);
    zero_vector<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vectemp, mat->width);
    zero_vector<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vectemp2, mat->width);

    i = 0;
    cont = 1;

    /* Calculate generalized eigenvector */
    while (cont == 1) {

        if (i%2 == 0) {
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_diag_inv, d_vec1, d_vectemp, d_no_offset, diag_inv->ndiags, diag_inv->width);
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_mat, d_vectemp, d_vectemp2, d_offset, mat->ndiags, mat->width);
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_diag_inv, d_vectemp2, d_vec2, d_no_offset, diag_inv->ndiags, diag_inv->width);
            add_shift<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, d_vec1, mat->width);
            zero_proj<<<1,1>>>();
            proj_inner<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, d_diag, mat->width);
            add_projection<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, d_vec1, d_diag, mat->width);
            zero_norm<<<1,1>>>();
            vec_norm<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, mat->width);
            manip_norm<<<1,1>>>();
            normalize<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, mat->width);
        }
        else {
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_diag_inv, d_vec2, d_vectemp, d_no_offset, diag_inv->ndiags, diag_inv->width);
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_mat, d_vectemp, d_vectemp2, d_offset, mat->ndiags, mat->width);
            diag_matvec_cuda<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_diag_inv, d_vectemp2, d_vec1, d_no_offset, diag_inv->ndiags, diag_inv->width);
            add_shift<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, d_vec2, mat->width);
            zero_proj<<<1,1>>>();
            proj_inner<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec2, d_diag, mat->width);
            add_projection<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, d_vec2, d_diag, mat->width);
            zero_norm<<<1,1>>>();
            vec_norm<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, mat->width);
            manip_norm<<<1,1>>>();
            normalize<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, mat->width);
        }

        norm_diff<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>(d_vec1, d_vec2, mat->width);

        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol( &diff, iter_diff, sizeof(iter_diff) );

        i++;

        if ( diff < tol || i > max_iter ) {
            cont = 0;
            calculate_eigenvalue_inner_prod<<< ceil( ( (float) mat->width ) / THREADSPERBLOCK ), THREADSPERBLOCK >>>( d_vec1, d_vec2, mat->width );
        }
    }


    cudaMemcpyFromSymbol( &lambda_max, d_lambda_max, sizeof(d_lambda_max) );

    if (i%2 == 0) {
        cudaMemcpy( vecout, d_vec2, mat->width * sizeof(float), cudaMemcpyDeviceToHost );
    }
    else {
        cudaMemcpy( vecout, d_vec1, mat->width * sizeof(float), cudaMemcpyDeviceToHost );
    }

    cudaFree( d_vec1 );
    cudaFree( d_vec2 );
    cudaFree( d_vectemp );
    cudaFree( d_vectemp2 );
    cudaFree( d_mat );
    cudaFree( d_diag );
    cudaFree( d_offset );
    cudaFree( d_no_offset );

    return vecout;
}

void apply_segmentation(
        image_t * input_im,
        float * seg,
        image_t * output_im1,
        image_t * output_im2)
{
    for (int i=0; i<input_im->height*input_im->width; i++) {
        if ( seg[i] > 0 ) {
            output_im1->red[i] = input_im->red[i];
            output_im1->green[i] = input_im->green[i];
            output_im1->blue[i] = input_im->blue[i];
            output_im2->red[i] = 0;
            output_im2->green[i] = 0;
            output_im2->blue[i] = 0;
        }
        else {
            output_im1->red[i] = 0;
            output_im1->green[i] = 0;
            output_im1->blue[i] = 0;
            output_im2->red[i] = input_im->red[i];
            output_im2->green[i] = input_im->green[i];
            output_im2->blue[i] = input_im->blue[i];
        }
    }
}
