#ifndef DIAG_MATVEC_H
#define DIAG_MATVEC_H

#include "diag_mat.h"

/**
 * @brief Wrapper for calling CUDA matvec operations
 *
 * @param mat Matix to apply
 * @param vec Input vector
 * @param vecout Output vector
 * @param n Length of vector
 */
float * diag_matvec(
    diag_mat * mat,
    float * vec,
    int n);

#endif
