/**
 * neuralNetwork.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix.h"
#include "vector.h"
#include "neuralNetwork.h"
#include "activation_fns.h"

//#define EMPTY_MATRIX {0, 0, NULL}

/* Initialize a neural network with the given structure */
void nn_init(Neural_Network *net, int numInputs, int numOutputs, int numHiddenLayers,
	  int *hiddenLayerSizes) {
  
  net->inputLayerSize = numInputs;
  net->outputLayerSize = numOutputs;
  net->numHiddenLayers = numHiddenLayers;
  net->hiddenLayerSizes = hiddenLayerSizes;

  //Need one weight matrix for input layer, then one for each hidden layer
  net->weights = malloc(sizeof(Matrix) * (1 + net->numHiddenLayers));
  
  nn_generateWeights(net);
}

/* Free the dynamically allocated memory in the neural network */
void nn_free(Neural_Network *net) {
  if(net) {
    if(net->weights) {
      int i;
      for(i = 0; i < net->numHiddenLayers + 1; i++) {
	matrix_free(&(net->weights[i]));
      }
    }
  }
}

/* Initialize a neural network with a pre-existing set of weights */
void nn_initWithWeights(Neural_Network *net, int numInputs, int numOutputs,
			int numHiddenLayers, int *hiddenLayerSizes, Matrix *weights) {
  int i;

  net->inputLayerSize = numInputs;
  net->outputLayerSize = numOutputs;
  net->numHiddenLayers = numHiddenLayers;
  net->hiddenLayerSizes = hiddenLayerSizes;

  //Need one weight matrix for input layer, then one for each hidden layer
  net->weights = malloc(sizeof(Matrix) * (1 + net->numHiddenLayers));

  //Copy the weights into the neural network
  for(i = 0; i < 1 + net->numHiddenLayers; i++) {
    net->weights[i] = emptyMatrix;
    matrix_copy(&(net->weights[i]), &weights[i]);
  }
}

/* Initialize the weight vectors with random floats */
void nn_generateWeights(Neural_Network *net) {
  int i, j, k, rows, cols;
  float randWeight, upperBound = 4.0;

  srand(time(NULL)); //COMMENT OUT FOR SAME SET OF RANDOM NUMBERS EACH TIME

  /* Weights for inputs --> hidden layer
   * Matrices need to be of size currentLayerNeurons x nextLayerNeurons
   */
  rows = net->inputLayerSize + 1; //Bottom row is for bias weights
  cols = net->hiddenLayerSizes[0];
  
  matrix_init(&(net->weights[0]), rows, cols);
  for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++) {
      //Generate a random weight in [-upperBound, upperBound]
      randWeight = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
      net->weights[0].m[i][j] = randWeight;
    }
  }
  
  //Weights for hidden layers --> hidden layers
  for(i = 0; i < net->numHiddenLayers - 1; i++) {
    rows = net->hiddenLayerSizes[i] + 1; //Bottom row is for bias weights
    cols = net->hiddenLayerSizes[i+1];
    matrix_init(&(net->weights[i+1]), rows, cols);
    for(j = 0; j < rows; j++) {
      for(k = 0; k < cols; k++) {	
	//Generate a random weight in [0, upperBound]
	randWeight = ((float)rand()/(float)(RAND_MAX/upperBound) * 2) - upperBound;
	net->weights[i+1].m[j][k] = randWeight;
      }
    }
  }

  //Weights for hidden layer --> outputlayer
  int finalIndex = net->numHiddenLayers - 1;
  rows = net->hiddenLayerSizes[finalIndex];
  cols = net->outputLayerSize;
  matrix_init(&(net->weights[finalIndex+1]), rows, cols);
  for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++) {
      //Generate a random weight in [-upperBound, upperBound]
      randWeight = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
      net->weights[finalIndex+1].m[i][j] = randWeight;
    }
  }
}

/* Forward propogate inputs through the network and return the output vector */
Matrix nn_forward(Neural_Network *net, Matrix *inputs) {
  int i;
  Matrix metro1, metro2; //The matrices that will traverse the network

  //Forward propagate---

  //First operate on input layer	       
  metro1 = matrix_multiply_slow(inputs, &(net->weights[0]));
  sigmoid_matrix(&metro1);
  
  //Now operate on hidden layers
  for(i = 0; i < net->numHiddenLayers; i++) {
    append_ones(&metro1);
    metro2 = matrix_multiply_slow(&metro1, &(net->weights[i+1]));
    sigmoid_matrix(&metro2);
    matrix_copy(&metro1, &metro2);
    matrix_free(&metro2);
    
    //Start with m1
    //m2 = m1 * weights
    //m1->weights = m2->weights
    //m2->weights == null
  }

  return metro1;
}

/* Forward propogate inputs through the network, return ouput and activity matrices
   from each layer */
Matrix nn_forward_activity(Neural_Network *net, Matrix *inputs, Matrix **zloc, Matrix **aloc) {
  int i;
  Matrix metro1, metro2; //The matrices that will traverse the network
  Matrix *z, *a;
  z = malloc(sizeof(Matrix) * net->numHiddenLayers + 1);
  a = malloc(sizeof(Matrix) * (net->numHiddenLayers));
  *zloc = z;
  *aloc = a;
  for(i = 0; i < net->numHiddenLayers + 1; i++) {
    //Matrix emptyMatrix = {0, 0, NULL};
    z[i] = emptyMatrix; //This may be a problem. If so, use macro instead
                             // #define EMPTY_MATRIX {0, 0, NULL}
    if(i != net->numHiddenLayers) {
      a[i] = emptyMatrix;
    }
  }

  //Forward propagate---
  
  //See modifications to nn_forward
  //Do not copy, but do not free
  //Reassign pointers and make other pointers null, be sure not to leak memory
  
  
  //First operate on input layer
  metro1 = matrix_multiply_slow(inputs, &(net->weights[0]));
  matrix_copy(&z[0], &metro1); 
  sigmoid_matrix(&metro1);
  matrix_copy(&a[0], &metro1);
  
  //Now operate on hidden layers
  for(i = 1; i < net->numHiddenLayers + 1; i++) {
    metro2 = matrix_multiply_slow(&metro1, &(net->weights[i]));
    matrix_copy(&z[i], &metro2);
    sigmoid_matrix(&metro2);
    if(i != net->numHiddenLayers) { //Do not copy output matrix into sigmoid activity
      matrix_copy(&a[i], &metro2);
    }
    matrix_copy(&metro1, &metro2);
  }
  
  return metro1;
}

/* Subtracts all gradients from the current weights of the neural network */
void nn_updateWeights(Neural_Network *net, Matrix *gradients, float learningRate) {
  int i, j, n;
  for(n = 0; n < net->numHiddenLayers + 1; n++) {
    for(i = 0; i < gradients[n].rows; i++) {
      for(j = 0; j < gradients[n].cols; j++) {
	net->weights[n].m[i][j] -= gradients[n].m[i][j] * learningRate;
      }
    }
  }
}
