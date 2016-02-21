/**
 * cnn_forwardPropTest.c
 * Jack Walpuck
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "matrix.h"
#include "conv_neuralNetwork.h"

int main(int argc, char *argv[]) {
  Conv_Neural_Network test;
  Matrix input, output;
  int i, j;
  int outputLayerSize = 2;
  int numHiddenLayers = 2;
  int numFullLayers = 2;
  int *fullLayerSizes, *kernelSizes, *depths;
  int stride = 1;
  int inputRows = 30, inputCols = 30;
  float upperBound = 255.0;

  fullLayerSizes = malloc(sizeof(int) * numFullLayers);
  kernelSizes = malloc(sizeof(int) * numHiddenLayers);
  depths = malloc(sizeof(int) * numHiddenLayers);
	
  fullLayerSizes[0] = 4;
  fullLayerSizes[1] = 2;
	
  kernelSizes[0] = 5;
  kernelSizes[1] = 3;
	
  depths[0] = 4;
  depths[1] = 12;

  //Manually create the input matrix
  matrix_init(&input, inputRows, inputCols);
  for(i = 0; i < inputRows; i++) {
    for(j = 0; j < inputCols; j++) {
      input.m[i][j] = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
    }
  }

  //Initialize the test network
  cnn_init(&test, outputLayerSize, numHiddenLayers, numFullLayers, fullLayerSizes, kernelSizes, depths, stride);

  /********** Display generated parameters ***************/
  printf("After initialization:\n");

  //Hidden layer data
  printf("Number of hidden layers: %d\n", test.numHiddenLayers);
  for(i = 0; i < test.numHiddenLayers; i++) {
    for(j = 0; j < test.depths[i]; j++) {
      printf("Kernel at layer %d, depth %d is %dx%d:\n", i, j, test.kernels[i][j].rows, test.kernels[i][j].cols);
      matrix_print(&test.kernels[i][j], stdout);
    }
    printf("\n");
  }
  printf("\n\n");

  printf("Number of fully connected layers: %d\n", test.numFullLayers);
  for(i = 0; i < test.numFullLayers+1; i++) {
    printf("Weight matrix %d is %dx%d:\n", i, test.fullLayerWeights[i].rows, test.fullLayerWeights[i].cols);
    matrix_print(&test.fullLayerWeights[i], stdout);
  }

  //Output layer data
  printf("Number of outputs: %d\n", test.outputLayerSize);

  /*********** Test Forward Propagation ****************/

  output = cnn_forward(&test, &input);
  printf("Output: \n");
  matrix_print(&output, stdout);

  matrix_free(&input);
  matrix_free(&output);
  cnn_free(&test);

  return 0;
}
