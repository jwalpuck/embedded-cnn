/**
 * cnn_backPropTest.c
 * Jack Walpuck
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "matrix.h"
#include "conv_neuralNetwork.h"
#include "training.h"

int main(int argc, char *argv[]) {
  Conv_Neural_Network test;
  Matrix input, output, correct_output, *dummy, **kernel_gradients, *FC_gradients;
  int i, j;
  int numClasses = 10;
  int numHiddenLayers = 2;
  int numFullLayers = 2;
  int *fullLayerSizes, *kernelSizes, *depths;
  int stride = 1;
  int inputRows = 10, inputCols = 10;
  float upperBound = 255.0;
  correct_output = emptyMatrix;

  fullLayerSizes = malloc(sizeof(int) * numFullLayers);
  kernelSizes = malloc(sizeof(int) * numHiddenLayers);
  depths = malloc(sizeof(int) * numHiddenLayers);

  fullLayerSizes[0] = 4;
  fullLayerSizes[1] = 2;
	
  kernelSizes[0] = 3;
  kernelSizes[1] = 3;
	
  depths[0] = 4;
  depths[1] = 12;

  //Randomly create the input matrix
  matrix_init(&input, inputRows, inputCols);
  for(i = 0; i < inputRows; i++) {
    for(j = 0; j < inputCols; j++) {
      input.m[i][j] = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - (upperBound / 2);
    }
  }

  //Create the output matrix
  matrix_init(&correct_output, 1, numClasses);
  correct_output.m[0][numClasses / 2] = 1;
  

  //Initialize the test network
  printf("Initializing network...\n");
  cnn_init(&test, numClasses, numHiddenLayers, numFullLayers, fullLayerSizes, kernelSizes, depths, stride);
  printf("Network initialized\n");

  /*************** Test back propagation *******************/
  //Allocate space for kernel gradients
  kernel_gradients = malloc(sizeof(Matrix *) * test.numHiddenLayers);
  for(i = 0; i < test.numHiddenLayers; i++) {
    kernel_gradients[i] = malloc(sizeof(Matrix) * test.depths[i]);
  }
  //Allocate space for fully connected layer gradients
  FC_gradients = malloc(sizeof(Matrix) * test.numFullLayers);

  //Test it
  printf("Performing back propagation\n");
  cnn_backProp(&test, &input, &correct_output, kernel_gradients, FC_gradients);
  
  printf("KERNEL GRADIENTS:\n");
  for(i = 0; i < test.numHiddenLayers; i++) {
    printf("LAYER %d\n", i);
    for(j = 0; j < test.depths[i]; j++) {
      matrix_print(&kernel_gradients[i][j], stdout);
    }
  }

  printf("FULLY CONNECTED WEIGHT GRADIENTS\n");
  for(i = 0; i < test.numFullLayers; i++) {
    matrix_print(&FC_gradients[i], stdout);
  }

  return 0;
}
