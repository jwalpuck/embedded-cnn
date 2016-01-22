/**
 * nn_forwardPropTest.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "vector.h"
#include "neuralNetwork.h"

int main(int argc, char *argv[]) {

  /* //DEBUG: Sigmoid test */
  /* float result = (float)1/(float)(1+pow(M_E, -1)); */
  /* printf("Sigmoid(1) = %.9f\n", result); */

  
  Neural_Network test;
  Matrix input, output;
  int numHiddenLayers = 1;
  int inputLayerSize = 2;
  int numInputTuples = 3;
  int outputLayerSize = 1;

  //Create input matrix, being sure to normalize
  matrix_init(&input, numInputTuples, inputLayerSize);
  input.m[0][0] = (float)3 / 10;
  input.m[0][1] = (float)5 / 5;
  input.m[1][0] = (float)5 / 10;
  input.m[1][1] = (float)1 / 5;
  input.m[2][0] = (float)10 / 10;
  input.m[2][1] = (float)2 / 5;
  /* input.m[0][0] = (float)3 ; */
  /* input.m[0][1] = (float)5; */
  /* input.m[1][0] = (float)5; */
  /* input.m[1][1] = (float)1; */
  /* input.m[2][0] = (float)10; */
  /* input.m[2][1] = (float)2; */
  
  //Assign hidden layer sizes
  int *hiddenLayerSizes = malloc(sizeof(int) * numHiddenLayers);
  hiddenLayerSizes[0] = 3;

  //Initialize the neural network with assigned parameters
  //nn_init(&test, inputLayerSize, outputLayerSize, numHiddenLayers, hiddenLayerSizes);
  
  /* output = nn_forward(&test, &input); */
  /* printf("Output: \n"); */
  /* matrix_print(&output, stdout); */

  //nn_free(&test);

  //Test with the results from the video
  Matrix *weights = malloc(sizeof(Matrix) * (numHiddenLayers + 1));
  matrix_init(&(weights[0]), 2, 3);
  weights[0].m[0][0] = 2.48568107;
  weights[0].m[0][1] = -0.81688985;
  weights[0].m[0][2] = -2.06193462;
  weights[0].m[1][0] = 0.20987088;
  weights[0].m[1][1] = -0.24268572;
  weights[0].m[1][2] = -0.07502123;

  matrix_init(&(weights[1]), 3, 1);
  weights[1].m[0][0] = 3.49996137;
  weights[1].m[1][0] = -1.20494324;
  weights[1].m[2][0] = -2.9143517;

  /* printf("Params: \n"); */
  /* matrix_print(&weights[0], stdout); */
  /* matrix_print(&weights[1], stdout); */
  
  nn_initWithWeights(&test, inputLayerSize, outputLayerSize, numHiddenLayers,
		     hiddenLayerSizes, weights);
  
  output = nn_forward(&test, &input);
  printf("Output: \n");
  matrix_print(&output, stdout);
  
  //Clean up
  matrix_free(&input);
  matrix_free(&output);
}
