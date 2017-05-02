
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

diag_mat * diag_sqrt(
    diag_mat* mat)
{
  diag_mat * main_diag = diag_alloc( mat->width, 1, mat->width, mat->height);

  main_diag->offset[0] = 0;

  for (int i=0; i<mat->ndiags; i++) {
    if ( mat->offset[i] == 0 ) {
      for (int j=0; j<mat->width; j++) {
        main_diag->vals[j] = sqrt(mat->vals[ i*mat->width + j]);
        main_diag->nnz++;
      }
    }
  }

  float sum = 0;
  for (int i=0; i<mat->width; i++) {
    sum += main_diag->vals[i] * main_diag->vals[i];
  }

  float factor = (float) (1.0 / (double) (sqrt(sum)) );

  for (int i=0; i<mat->width; i++) {
    main_diag->vals[i] = main_diag->vals[i] * factor;
  }


  return main_diag;
}

diag_mat * diag_inv_sqrt(
    diag_mat* mat)
{
  diag_mat * main_diag = diag_alloc( mat->width, 1, mat->width, mat->height);

  main_diag->offset[0] = 0;

  for (int i=0; i<mat->ndiags; i++) {
    if ( mat->offset[i] == 0 ) {
      for (int j=0; j<mat->width; j++) {
        main_diag->vals[j] = 1.0 / sqrt(mat->vals[ i*mat->width + j]);
        main_diag->nnz++;
      }
    }
  }

  return main_diag;
}

void diag_free(
    diag_mat * diag)
{
  free(diag->vals);
  free(diag->offset);
  free(diag);
}
