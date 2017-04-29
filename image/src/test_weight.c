
#include <stdlib.h>

#include "image.h"
#include "csr_mat.h"
#include "diag_mat.h"
#include "matvec.h"

int main(
    int argc,
    char ** argv)
{

  /* Artificial test */
  image_t * im = image_alloc(10, 10);

  for (int i=0; i<10; i++) {
    for (int j=0; j<10; j++) {
      im->red[i*10 + j] = i*10+j;
    }
  }

  csr_mat * wgt = create_weight_csr(im, 3);

  diag_mat * wgt_diag = create_weight_diag(im, 2);

  /* Test on image */

  image_t * im2 = image_load("karypis.bmp");
  image_t * gray = grayscale(im);

  csr_mat * wgt2 = create_weight_csr(im2, 2.5);

  /* Test matrix multiplication */
  diag_mat * mat = diag_alloc(20, 2, 10, 10);
  mat->offset[0] = 0;
  mat->offset[1] = -2;
  mat->nnz = 20;

  for (int i=0; i<10; i++) {
    mat->vals[i] = 2;
    mat->vals[i+10] = 1;
  }

  float vec[10];

  for (int i=0; i<10; i++) {
    vec[i] = i;
  }

  float * vecout = diag_matvec( mat, vec, 10 );

  free(im);
  free(wgt);
  free(wgt2);
  free(wgt_diag);

  return EXIT_SUCCESS;

}
