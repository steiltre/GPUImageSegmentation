#ifndef CSR_H
#define CSR_H



/**
 * @brief Structure for compressed sparse row format
 */
typedef struct
{
  /** Pointer to beginning or rows */
  unsigned * ptr;

  /** Column indices */
  unsigned * cols;

  /** Matrix values */
  float * vals;
  /* Flags for segmented scan*/
  unsigned *flags;
  /** Number of nonzeros */
  int nnz;
  /* Dim of matrix*/
  int rows;
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

#endif
