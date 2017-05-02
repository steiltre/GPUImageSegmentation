#ifndef POWERIT_H
#define POWERIT_H

//#include <cuda_runtime.h>
#include "csr_mat.h"


/* 
    Taken from Nvidia cuda examples
    required for reduction kernel
*/

void eigenvalue_solver(csr_mat *h_matrix, float *h_vec,float *h_diag);

#endif
