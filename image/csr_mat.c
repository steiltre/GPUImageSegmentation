
#include <stdlib.h>
#include <stdio.h>

#include "csr_mat.h"

csr_mat * csr_alloc(
    int max_nnz,
    int rows)
{
  csr_mat * csr = malloc(sizeof(*csr));

  csr->ptr = malloc( (rows+1) * sizeof(*(csr->ptr)) );
  csr->cols = malloc( max_nnz * sizeof(*(csr->cols)) );
  csr->vals = malloc( max_nnz * sizeof(*(csr->vals)) );

  csr->nnz = 0;

  return csr;
}
