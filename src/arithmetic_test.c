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

int main(int argc, char *argv[]) {
  int i, j;
  Neural_Network test;
  int numHiddenLayers = 4; //4
  int inputLayerSize = 2; //+1 for bias
  int outputLayerSize = 1;
  int numInputTuples;
  float initialCost, finalCost, testInitialCost, testFinalCost, learningRate, *yhat;
  Matrix input, input2, expected_output, expected_output2, *gradient, output1, output2, testOutput;
  char *trainingFileName, *testFileName;
	
	if(argc < 3) {
		printf("Usage: file_parserTest <training fileName> <test fileName\n:");
		return -1;
	}
	
	trainingFileName = argv[1];
	testFileName = argv[2];
	file_to_matrix(trainingFileName, &input, &expected_output);
	file_to_matrix(testFileName, &input2, &expected_output2);
	
// 	printf("\n\nMatrices:\n");
// 	printf("Input1:\n");
// 	matrix_print(&input, stdout);
// 	printf("Output1:\n");
// 	matrix_print(&expected_output, stdout);
// 	printf("Input2\n");
// 	matrix_print(&input2, stdout);
// 	printf("Output2\n");
// 	matrix_print(&expected_output2, stdout);

	printf("Read in matrices\n");

  learningRate = 0.1; //0.3

	matrix_normalize_columns(&input);
	matrix_normalize_columns(&input2);
	
	//Normalize output
	matrix_normalize_columns(&expected_output);
	matrix_normalize_columns(&expected_output2);

  //Assign hidden layer sizes
  int *hiddenLayerSizes = malloc(sizeof(int) * numHiddenLayers);
  hiddenLayerSizes[0] = 3;
  hiddenLayerSizes[1] = 4;
  hiddenLayerSizes[2] = 8;
  hiddenLayerSizes[3] = 2;

  /**** Test generating random weights ****/
  //Initialize the neural network with assigned parameters
  numInputTuples = input.rows;
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
  
  printf("Checking output\n");

  output1 = nn_forward(&test, &input);
  
  printf("Output checked\n");
  
  //Look at initial cost
  initialCost = cost_fn(&test, &input, &expected_output);
  testInitialCost = cost_fn(&test, &input2, &expected_output2);
  
    printf("%%%%%%Original weights:\n");
    for(j = 0; j < 4; j++) {
    	matrix_print(&(test.weights[j]), stdout);
    } 

  for(i = 0; i < 100000; i++) { //100000
  	//printf("Computing gradient %d\n", i);
    //Compute numerical gradient
    gradient = cost_fn_prime(&test, &input, &expected_output);
    
//     printf("%%%%%%Original weights:\n");
//     for(j = 0; j < 4; j++) {
//     	matrix_print(&(test.weights[j]), stdout);
//     }   
    
    //Modify the weights according to the gradient
    nn_updateWeights(&test, gradient, learningRate);
    
//     printf("%%%%%%Updated weights:\n");
//     for(j = 0; j < 4; j++) {
//     	matrix_print(&(test.weights[j]), stdout);
//     }

    matrix_free(gradient); //Prevent memory leak
  }

  //Compute the cost of the training data
  finalCost = cost_fn(&test, &input, &expected_output);
  
  //Compute the output and cost of the test data
	testOutput = nn_forward(&test, &input2);
	testFinalCost = cost_fn(&test, &input2, &expected_output2);
	
	//Compute numerical yhat
	yhat = malloc(sizeof(float) * testOutput.rows);
	for(i = 0; i < testOutput.rows; i++) {
		yhat[i] = fabsf(expected_output2.m[i][0] - testOutput.m[i][0]);
	}

  //Print the results
  printf("Initial (training) cost: %f; Final (training) cost: %f\n", initialCost, finalCost);
  printf("Initial (test) cost: %f; Final (test) cost: %f\n", testInitialCost, testFinalCost);

  //*Optional: Show output
  output2 = nn_forward(&test, &input);
  printf("With initial (random) weights: \n");
  matrix_print(&output1, stdout);
  printf("With optimized weights: \n");
  matrix_print(&output2, stdout);

	printf("\n\n\n\nTraining answers:\n");
	matrix_print(&expected_output2, stdout);
	printf("Output answers\n");
	matrix_print(&testOutput, stdout);
  
  //Show yhat for the test data
  printf("\ny_hat for the test data:\n");
  for(i = 0; i < testOutput.rows; i++) {
  	printf("%f\n", yhat[i]);
  }
  
  //Free the neural network
  nn_free(&test);
  matrix_free(&input);
  matrix_free(&expected_output);
  matrix_free(&output1);
  matrix_free(&output2);
  
  //Free local variables
  free(yhat);
}
