#ifndef DIAG_H
#define DIAG_H

/**
 * @brief Structure for sparse matrix in diagonal format
 */
typedef struct
{
  /** Array of data values */
  float * vals;

  /** Offsets of diagonals */
  int * offset;

  /** Matrix dimensions */
  int width;
  int height;

  /** Number of nonzeros */
  unsigned nnz;
} diag_mat;

/**
 * @brief Allocate structures for diag matrix
 *
 * @param max_nnz Maximum number of nonzeros to be placed in matrix
 * @param width Number of columns in matrix
 * @param height Number of rows in matrix
 *
 * @return Initialized diag matrix
 */
diag_mat * diag_alloc(
    unsigned max_nnz,
    unsigned ndiags,
    int width,
    int height);

/**
 * @brief Free memory from diag matrix
 *
 * @param mat Matrix to free
 */
void diag_free(
    diag_mat * diag);

#endif
