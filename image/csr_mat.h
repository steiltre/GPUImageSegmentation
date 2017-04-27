#ifndef CSR_H
#define CSR_H

/**
 * @brief Structure for compressed sparse row format
 */
typedef struct
{
  /** Pointer to beginning or rows */
  int * ptr;

  /** Column indices */
  int * cols;

  /** Matrix values */
  float * vals;

  /** Number of nonzeros */
  int nnz;
} csr_mat;

/**
 * @brief Allocate structures for csr matrix
 *
 * @param max_nnz Maximum number of nonzeros to be placed in matrix
 * @param rows Number of rows
 *
 * @return Initialized csr matrix
 */
csr_mat * csr_alloc(
    int max_nnz,
    int rows);

/**
 * @brief Free memory from csr matrix
 *
 * @param mat Matrix to free
 */
void csr_free(
    csr_mat * mat);

#endif
