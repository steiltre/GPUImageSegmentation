#ifndef POWERIT_H
#define POWERIT_H

#include <cudpp.h>
#include <cuda_runtime.h>
#include "csr_mat.h"


/* 
    Taken from Nvidia cuda examples
    required for reduction kernel
*/

extern "C" void eigenvalue_solver(csr_mat *h_matrix, float *h_vec);

#endif