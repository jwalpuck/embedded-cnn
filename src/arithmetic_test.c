/**
 * arithmetic_test.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "vector.h"
#include "neuralNetwork.h"
#include "training.h"
#include "file_parser.h"
#include "my_timing.h"

int main(int argc, char *argv[]) {
  int i, j, selection, n;
  Neural_Network test;
  int numHiddenLayers = 8;
  int inputLayerSize = 2;
  int outputLayerSize = 1;
  double t1, t2, dt;
  float initialCost, finalCost, testInitialCost, testFinalCost, learningRate, avg_yhat1, avg_yhat2;
  Matrix input, input2, expected_output, expected_output2, *gradient, output1, output2, testOutput2;
  char *trainingFileName, *testFileName;
	
  if(argc < 5) {
    printf("Usage: file_parserTest <training fileName> <test fileName> <training option> <numIter> \n:");
    printf("Training options:\n");
    printf("0: Batch gradient descent\n");
    printf("1: Stochastic gradient descent\n");
    printf("2: Stochastic gradient descent with momentum\n");
    return -1;
  }
	
  trainingFileName = argv[1];
  testFileName = argv[2];
  selection = atoi(argv[3]);
  n = atoi(argv[4]);
  file_to_matrix(trainingFileName, &input, &expected_output);
  file_to_matrix(testFileName, &input2, &expected_output2);

  learningRate = 0.1; //0.3

  matrix_normalize_columns(&input);
  matrix_normalize_columns(&input2);
	
  //Normalize output
  matrix_normalize_columns(&expected_output);
  matrix_normalize_columns(&expected_output2);

  //Assign hidden layer sizes
  int *hiddenLayerSizes = malloc(sizeof(int) * numHiddenLayers);
  hiddenLayerSizes[0] = 2;
  hiddenLayerSizes[1] = 3;
  hiddenLayerSizes[2] = 2;
  hiddenLayerSizes[3] = 8;
  hiddenLayerSizes[4] = 4;
  hiddenLayerSizes[5] = 6;
  hiddenLayerSizes[6] = 7;
  hiddenLayerSizes[7] = 5;

  /**** Test generating random weights ****/
  //Initialize the neural network with assigned parameters
  nn_init(&test, inputLayerSize, outputLayerSize, numHiddenLayers, hiddenLayerSizes);
  
  //Look at initial cost
  initialCost = cost_fn(&test, &input, &expected_output);
  testInitialCost = cost_fn(&test, &input2, &expected_output2);

  //Compute arithmetic error
  avg_yhat1 = 0;
  avg_yhat2 = 0;
  output2 = nn_forward(&test, &input);
  testOutput2 = nn_forward(&test, &input2);
  for(j = 0; j < expected_output.rows; j++) {
    avg_yhat1 += fabsf(expected_output.m[j][0] - output2.m[j][0]);
    avg_yhat2 += fabsf(expected_output2.m[j][0] - testOutput2.m[j][0]);
  }
  matrix_free(&output2);
  matrix_free(&testOutput2);
  avg_yhat1 /= expected_output.rows;
  avg_yhat2 /= expected_output2.rows;
  printf("\nInitial arithmetic error for training data: %f(%.02f%%)\n", avg_yhat1, avg_yhat1 * 100);
  printf("Initial arithmetic error for test data: %f(%.02f%%)\n\n", avg_yhat2, avg_yhat2 * 100);
  
  if(selection == 0) { //Batch gradient descent
    t1 = get_time_sec();
    for(i = 0; i < n; i++) {

      //Compute numerical gradient
      gradient = cost_fn_prime(&test, &input, &expected_output); 
		
      //Modify the weights according to the gradient

      //Clean up old gradients before new ones are written
      for(j = 0; j < numHiddenLayers+1; j++) {
	matrix_free(&gradient[j]);
      }
      free(gradient);
    }
    t2 = get_time_sec();
  }
  else if(selection == 1) { //Stochastic gradient descent
    t1 = get_time_sec();
    stochastic_grad_descent(&test, &input, &expected_output, n, learningRate);
    t2 = get_time_sec();
  }
  else if(selection == 2) { //Stochastic gradient descent with momentum
    t1 = get_time_sec();
    stochastic_grad_descent_momentum(&test, &input, &expected_output, n, learningRate, 0.5);
    t2 = get_time_sec();
  }

  //Compute the cost of the training data
  finalCost = cost_fn(&test, &input, &expected_output);
  
  //Compute the output and cost of the test data
  testOutput2 = nn_forward(&test, &input2);
  testFinalCost = cost_fn(&test, &input2, &expected_output2);
  
  //Compute arithmetic error
  avg_yhat1 = 0;
  avg_yhat2 = 0;
  output2 = nn_forward(&test, &input);
  testOutput2 = nn_forward(&test, &input2);
  for(j = 0; j < expected_output.rows; j++) {
    avg_yhat1 += fabsf(expected_output.m[j][0] - output2.m[j][0]);
    avg_yhat2 += fabsf(expected_output2.m[j][0] - testOutput2.m[j][0]);
  }
  matrix_free(&output2);
  matrix_free(&testOutput2);
  avg_yhat1 /= expected_output.rows;
  avg_yhat2 /= expected_output2.rows;
  printf("Final arithmetic error for training data: %f(%.02f%%)\n", avg_yhat1, avg_yhat1 * 100);
  printf("Final arithmetic error for test data: %f(%.02f%%)\n\n", avg_yhat2, avg_yhat2 * 100);

  printf("Initial (training) cost: %f; Final (training) cost: %f\n", initialCost, finalCost);
  printf("Initial (test) cost: %f; Final (test) cost: %f\n\n", testInitialCost, testFinalCost);
  
  //Show timing
  dt = t2-t1;
  printf("Runtime for strategy %d: %.04f seconds\n\n", selection, dt);
  
  //Clean up
  nn_free(&test);
  matrix_free(&input);
  matrix_free(&input2);
  matrix_free(&expected_output);
  matrix_free(&expected_output2);
  free(hiddenLayerSizes);
 
  return 0;
}
