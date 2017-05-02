#ifndef IMAGE_SEGMENTATION_H
#define IMAGE_SEGMENTATION_H

#include "diag_mat.h"
#include "image.h"

/**
 * @brief Segment based on image segmentation algorithm
 *
 * @param mat Diagonal format sparse matrix to segment
 */
float * segmentation(
    diag_mat * mat);


/**
 * @brief Apply segmentation vector to image
 *
 * @param input_im  Image being segmented
 * @param seg  Segmentation vector to be applied
 * @param(out) output_im1  One cluster of seg applied to input_im
 * @param(out) output_im2  Other cluster of seg applied to input_im
 */
void apply_segmentation(
    image_t * input_im,
    float * seg,
    image_t * output_im1,
    image_t * output_im2);

#endif
