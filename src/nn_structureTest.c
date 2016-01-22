/**
 * nn_structureTest.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "vector.h"
#include "neuralNetwork.h"

int main(int argc, char *argv[]) {
  int i;
  Neural_Network test;
  int numHiddenLayers = 1; //4
  int inputLayerSize = 2;
  int outputLayerSize = 1;

  //Assign hidden layer sizes
  int *hiddenLayerSizes = malloc(sizeof(int) * numHiddenLayers); //Size 1 for test
  hiddenLayerSizes[0] = 3;
  /* hiddenLayerSizes[1] = 4; */
  /* hiddenLayerSizes[2] = 8; */
  /* hiddenLayerSizes[3] = 2; */

  /**** Test generating random weights ****/
  //Initialize the neural network with assigned parameters
  nn_init(&test, inputLayerSize, outputLayerSize, numHiddenLayers, hiddenLayerSizes);

  printf("After initialization:\n\n");

  //Input data
  printf("Number of inputs: %d\n", test.inputLayerSize);  
  printf("Input layer weights:\n");
  matrix_print(&test.weights[0], stdout);
  printf("\n\n");

  //Hidden layer data
  printf("Number of hidden layers: %d\n", test.numHiddenLayers);
  for(i = 0; i < test.numHiddenLayers; i++) {
    printf("Hidden Layer %d: \n", i);
    matrix_print(&test.weights[i+1], stdout);
    printf("\n");
  }
  printf("\n\n");

  //Output layer data
  printf("Number of outputs: %d\n", test.outputLayerSize);

  //Free the neural network
  nn_free(&test);
  
  /**** Test using custom weights ****/
  Matrix *weights = malloc(sizeof(Matrix) * (numHiddenLayers + 1));
  matrix_init(&(weights[0]), 2, 3);
  weights[0].m[0][0] = -2.48772788e-09;
  weights[0].m[0][1] = 1.62507244e-08;
  weights[0].m[0][2] = -9.32119968e-08;
  weights[0].m[1][0] = -1.32655986e-09;
  weights[0].m[1][1] = 3.50716727e-08;
  weights[0].m[1][2] = -1.65122302e-07;

  matrix_init(&(weights[1]), 3, 1);
  weights[1].m[0][0] = -2.14796459e-07;
  weights[1].m[1][0] = -2.05463705e-07;
  weights[1].m[2][0] = 7.32973302e-08;

  nn_initWithWeights(&test, inputLayerSize, outputLayerSize, numHiddenLayers,
		     hiddenLayerSizes, weights);

  printf("Predefined weights:");
  for(i = 0; i < test.numHiddenLayers + 1; i++) {
    matrix_print(&test.weights[i], stdout);
    printf("\n");
  }
  
  
  //Clean up
  nn_free(&test);
  free(hiddenLayerSizes);
}
