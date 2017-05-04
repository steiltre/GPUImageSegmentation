

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "power_iteration.h"
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
	
 
  int dim = im->height * im->width;
  csr_mat *diag_csr = csr_alloc(dim,dim);
  csr_mat *wgt_csr = create_weight_csr(gray,atoi(argv[2]),diag_csr);
 
  float * h_vec = malloc(sizeof(float)*dim);
  int i;
  printf("In main with dim = %d\n",dim);
  for(i = 0 ; i < dim; i++){
	h_vec[i] = 1;
  }

 
  eigenvalue_solver(wgt_csr,h_vec,diag_csr->vals);
  

  float * seg_vec = segmentation(wgt_diag);
  

  image_t *out_im1 = image_alloc(gray->width, gray->height);
  image_t *out_im2 = image_alloc(gray->width, gray->height);

  image_t *out_im3 = image_alloc(gray->width, gray->height);
  image_t *out_im4 = image_alloc(gray->width, gray->height);	

  apply_segmentation( gray, seg_vec, out_im1, out_im2 );

  apply_segmentation( gray, h_vec, out_im3, out_im4 );

  /*
  printf("Testing signs\n");
  for(i = 0; i < dim; i++){
    printf("%f - %f\n",seg_vec[i],h_vec[i]);
    if(seg_vec[i]*h_vec[i] <= 0){
      printf("Sign mismatch at %d\n",i);
    }
  }
  */

  image_write_bmp("seg1.bmp", out_im1);
  image_write_bmp("seg2.bmp", out_im2);

  image_write_bmp("seg3.bmp", out_im3);
  image_write_bmp("seg4.bmp", out_im4);

 free(h_vec);
}
