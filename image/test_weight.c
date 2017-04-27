
#include <stdlib.h>

#include "image.h"
#include "csr_mat.h"
#include "diag_mat.h"

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

  return EXIT_SUCCESS;

}
