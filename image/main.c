


/*
 * For high-resolution timers.
 */
#ifndef _POSIX_C_SOURCE
  #define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>

#include "image.h"



/*
 * Stencils.
 */
float identity[3][3] = {
  {0.,  0.,  0.},
  {0.,  1.,  0.},
  {0.,  0.,  0.}
};

float blur[3][3] = {
  {0.111,  0.111,  0.111},
  {0.111,  0.111,  0.111},
  {0.111,  0.111,  0.111}
};


float sharp[3][3] = {
  { 0.0,  -1.0,   0.0},
  {-1.0,   5.0,  -1.0},
  { 0.0,  -1.0,   0.0},
};


float smooth[3][3] = {
  {1.0,  2.0,  1.0},
  {2.0,  4.0,  2.0},
  {1.0,  2.0,  1.0},
};

float emboss[3][3] = {
  {2.,  0.,  0.},
  {0., -1.,  0.},
  {0.,  0., -1.}
};


float dither[3][3] = {
  {6.,  8.,  4.},
  {1.,  0.,  3.},
  {5.,  2.,  7.}
};



int main(
    int argc,
    char * * argv)
{
  if(argc < 3) {
    fprintf(stderr, "usage: %s <input-image.{jpg,bmp,png}> <output-image.bmp>\n", *argv);
    return EXIT_FAILURE;
  }

  image_t * im = image_load(argv[1]);

  image_t * output = grayscale(im);

  image_write_bmp(argv[2], output);

  image_free(im);
  image_free(output);

  return EXIT_SUCCESS;
}



