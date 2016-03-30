/**
 * cnn_mnistTest.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "conv_neuralNetwork.h"
#include "training.h"
#include "my_timing.h"
#include "mnist_parser.h"

int main(int argc, char *argv[]) {
  Conv_Neural_Network ataraxy;
  int tr_n, ts_n, numClasses, numHiddenLayers, numFullLayers, stride, *fullLayerSizes, *kernelSizes, *depths, i;
  float initial_err, final_err;
  double t1, t2;
  Matrix *tr_images, *tr_labels, *ts_images, *ts_labels;
  char *tr_img_filename, *tr_lbl_filename, *ts_img_filename, *ts_lbl_filename;

  tr_n = 0;
  ts_n = 0;
  
  if(argc < 5) {
    printf("Usage: %s <mnist training image filepath> <mnist training label filepath> <mnist test image filepath> <mnist test label filepath>\n", argv[0]);
    exit(-1);
  }

  /***************** Parse images and labels ***************************************/
  tr_img_filename = argv[1];
  tr_lbl_filename = argv[2];
  ts_img_filename = argv[3];
  ts_lbl_filename = argv[4];

  tr_images = read_mnist(tr_img_filename, &tr_n);
  printf("%d training images read\n", tr_n);

  tr_labels = read_mnist_labels(tr_lbl_filename, &tr_n);
  printf("%d training labels read\n", tr_n);

  ts_images = read_mnist(ts_img_filename, &ts_n);
  printf("%d test images read\n", ts_n);

  ts_labels = read_mnist_labels(ts_lbl_filename, &ts_n);
  printf("%d test labels read\n", ts_n);

  /***************** Normalize the inputs ****************************/
  for(i = 0; i < tr_n; i++) {
    matrix_normalize_all(&tr_images[i]);
  }
  for(i = 0; i < ts_n; i++) {
    matrix_normalize_all(&ts_images[i]);
  }

  /****************** Initialize the network *************************/
  numClasses = 10;
  numHiddenLayers = 2;
  numFullLayers = 2;
  stride = 1;

  fullLayerSizes = malloc(sizeof(int) * numFullLayers);
  kernelSizes = malloc(sizeof(int) * numHiddenLayers);
  depths = malloc(sizeof(int) * numHiddenLayers);

  fullLayerSizes[0] = 4;
  fullLayerSizes[1] = 2;

  kernelSizes[0] = 5;
  kernelSizes[1] = 3;

  depths[0] = 4;
  depths[1] = 8;

  cnn_init(&ataraxy, numClasses, numHiddenLayers, numFullLayers, fullLayerSizes, kernelSizes, depths, stride);

  printf("Network initialized\n");

  /***************** Test ****************************************/
  //Calculate initial error
  initial_err = cross_entropy(&ataraxy, ts_images, ts_labels, ts_n);
  printf("Initial error: %f\n", initial_err);

  //SGD
  printf("Performing stochastic gradient descent...\n");
  t1 = get_time_sec();
  cnn_stochastic_grad_descent(&ataraxy, tr_images, tr_labels, 1, tr_n, 0.1);
  t2 = get_time_sec();
  printf("Stochastic gradient descent complete\n");

  //Calculate final error
  final_err = cross_entropy(&ataraxy, tr_images, tr_labels, ts_n);
  //final_err = 0;
  printf("Final error: %f\n", final_err);

  printf("Training runtime: %f seconds\n", t2-t1);

  return 0;
}
