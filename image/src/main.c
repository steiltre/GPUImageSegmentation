

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "image.h"
#include "image_segmentation.h"
#include "diag_mat.h"

int main(
    int argc,
    char ** argv)
{

  /* Call output using "test image.bmp n"
   * where image.bmp is the image to be segmented
   * and n is the number of neighbor pixels to use */

  image_t * im = image_load(argv[1]);

  image_t * gray = grayscale(im);

  diag_mat * wgt_diag = create_weight_diag(gray, atoi(argv[2]) );

  float * seg_vec = segmentation(wgt_diag);

  image_t *out_im1 = image_alloc(gray->width, gray->height);
  image_t *out_im2 = image_alloc(gray->width, gray->height);

  apply_segmentation( gray, seg_vec, out_im1, out_im2 );

  image_write_bmp("seg1.bmp", out_im1);
  image_write_bmp("seg2.bmp", out_im2);
}
