/**
 * gradient_test.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "vector.h"
#include "neuralNetwork.h"
#include "training.h"

int main(int argc, char *argv[]) {
  int i;
  Neural_Network test;
  int numHiddenLayers = 1; //4
  int inputLayerSize = 2 + 1; //+1 for bias
  int outputLayerSize = 1;
  int numInputTuples = 3;
  float initialCost, finalCost;
  Matrix input, expected_output, *gradient, output1, output2;

  //Create input matrix, being sure to normalize
  matrix_init(&input, numInputTuples, inputLayerSize);
  input.m[0][0] = (float)3 / 10;
  input.m[0][1] = (float)5 / 5;
  input.m[0][2] = (float)1; //Bias
  input.m[1][0] = (float)5 / 10;
  input.m[1][1] = (float)1 / 5;
  input.m[1][2] = (float)5 / 5; //Bias
  input.m[2][0] = (float)10 / 10;
  input.m[2][1] = (float)2 / 5;
  input.m[2][2] = (float)1; //Bias
  

  //Create expected output matrix
  matrix_init(&expected_output, numInputTuples, 1);
  expected_output.m[0][0] = 0.75;
  expected_output.m[1][0] = 0.82;
  expected_output.m[2][0] = 0.93;

  //Assign hidden layer sizes
  int *hiddenLayerSizes = malloc(sizeof(int) * numHiddenLayers); //Size 1 for test
  hiddenLayerSizes[0] = 3;
  /* hiddenLayerSizes[1] = 4; */
  /* hiddenLayerSizes[2] = 8; */
  /* hiddenLayerSizes[3] = 2; */

  /**** Test generating random weights ****/
  //Initialize the neural network with assigned parameters
  nn_init(&test, inputLayerSize, outputLayerSize, numHiddenLayers, hiddenLayerSizes);
  //printf("After initialization:\n\n");

  //Input data
  /* printf("Number of inputs: %d\n", test.inputLayerSize);   */
  /* printf("Input layer weights:\n"); */
  /* matrix_print(&test.weights[0], stdout); */
  /* printf("\n\n"); */

  /* //Hidden layer data */
  /* printf("Number of hidden layers: %d\n", test.numHiddenLayers); */
  /* for(i = 0; i < test.numHiddenLayers; i++) { */
  /*   printf("Hidden Layer %d: \n", i); */
  /*   matrix_print(&test.weights[i+1], stdout); */
  /*   printf("\n"); */
  /* } */
  /* printf("\n\n"); */

  /* //Output layer data */
  /* printf("Number of outputs: %d\n", test.outputLayerSize); */

  output1 = nn_forward(&test, &input);
  
  //Look at initial cost
  initialCost = cost_fn(&test, &input, &expected_output);

  for(i = 0; i < 100000; i++) {
    //printf("i = %d\n", i);
    //Compute numerical gradient
    gradient = cost_fn_prime(&test, &input, &expected_output);

    /* printf("$$$$$$$$$$$ GRADIENT $$$$$$$$$$$$\n"); */
    /* for(i = 0; i < test.numHiddenLayers + 1; i++) { */
    /*   printf("W%d\n", i+1); */
    /*   matrix_print(&gradient[i], stdout);  */
    /* } */

    //Modify the weights according to the gradient
    nn_updateWeights(&test, gradient);
    matrix_free(gradient); //Prevent memory leak
  }

  //Compute the new cost
  finalCost = cost_fn(&test, &input, &expected_output);

  //Print the results
  printf("Initial cost: %f; Final cost: %f\n", initialCost, finalCost);

  //*Optional: Show output
  output2 = nn_forward(&test, &input);
  printf("With initial (random) weights: \n");
  matrix_print(&output1, stdout);
  printf("With optimized weights: \n");
  matrix_print(&output2, stdout);
  
  //Free the neural network
  nn_free(&test);
  matrix_free(&input);
  matrix_free(&expected_output);
  matrix_free(&output1);
  matrix_free(&output2);
}
