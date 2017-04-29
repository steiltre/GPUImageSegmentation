
#define THREADSPERBLOCK 256
#define WARPLENGTH 32

#include <stdio.h>
#include <math.h>

extern "C"
{
#include "diag_mat.h"
#include "matvec.h"
}

/**
  @brief Sparse matvec using diag format

  @param d_mat Matrix of diagonal values
  @param d_vec Vector to multiply
  @param d_vecout Output vector
  @param d_offset Array of offsets to use for diagonals
  @param ndiags Number of diagonals
  @param n Length of d_vec
  */
__global__ void diag_matvec_cuda(
        float * d_mat,
        float *d_vec,
        float *d_vecout,
        int * d_offset,
        unsigned ndiags,
        unsigned n)
{

    float temp;

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        for (int i=0; i < ndiags; i++) {
            if (d_offset[i] + row < n && d_offset[i] + row >= 0) {
                temp += d_mat[ i*n + row ] * d_vec[ d_offset[i] + row];
            }
        }
    }

    d_vecout[row] = temp;
}

float * diag_matvec(
        diag_mat * mat,
        float * vec,
        int n)
{

  float *d_mat, *d_vec, *d_vecout;
  int *d_offset;

  float * vecout = (float*) malloc( n * sizeof(*vecout) );

  cudaMalloc( (void**) &d_mat, mat->nnz*sizeof(float) );
  cudaMalloc( (void**) &d_vec, n*sizeof(float) );
  cudaMalloc( (void**) &d_vecout, n*sizeof(float) );
  cudaMalloc( (void**) &d_offset, mat->ndiags*sizeof(int) );

  cudaMemcpy( d_mat, mat->vals, mat->nnz*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_vec, vec, n*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_offset, mat->offset, mat->ndiags*sizeof(int), cudaMemcpyHostToDevice );

  diag_matvec_cuda<<<1,10>>>( d_mat, d_vec, d_vecout, d_offset, mat->ndiags, n);

  cudaMemcpy( vecout, d_vecout, n*sizeof(float), cudaMemcpyDeviceToHost );

  return vecout;

}

