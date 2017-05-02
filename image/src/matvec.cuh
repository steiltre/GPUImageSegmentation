

__global__ void diag_matvec_cuda(
    float * d_mat,
    float * d_vec,
    float * d_vecout,
    int * d_offset,
    unsigned ndiags,
    unsigned n);
