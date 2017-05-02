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

  /** Number of diagonals */
  unsigned ndiags;

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
 * @brief Extract square root of main diagonal
 *
 * @param mat Matrix to extract diagonal from
 *
 * @return Square root of diagonal entries
 */
diag_mat * diag_sqrt(
    diag_mat * mat);


/**
 * @brief Extract inverse square root of main diagonal
 *
 * @param mat Matrix to extract diagonal from
 *
 * @return Diagonal matrix containing square root of main diagonal in original matrix
 */
diag_mat * diag_inv_sqrt(
    diag_mat * mat);

/**
 * @brief Free memory from diag matrix
 *
 * @param mat Matrix to free
 */
void diag_free(
    diag_mat * diag);

#endif
