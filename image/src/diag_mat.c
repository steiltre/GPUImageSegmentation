
#include <stdlib.h>
#include <stdio.h>

#include "diag_mat.h"

diag_mat * diag_alloc(
    unsigned max_nnz,
    unsigned ndiags,
    int width,
    int height)
{
  diag_mat * diag = malloc( sizeof(*diag) );

  diag->vals = malloc( max_nnz * sizeof(*(diag->vals)) );
  diag->offset = malloc( ndiags * sizeof(*(diag->offset)) );

  diag->width = width;
  diag->height = height;

  diag->ndiags = ndiags;
  diag->nnz = 0;

  return diag;
}

void diag_free(
    diag_mat * diag)
{
  free(diag->vals);
  free(diag->offset);
  free(diag);
}
