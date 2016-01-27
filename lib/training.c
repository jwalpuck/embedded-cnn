/**
 * training.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "neuralNetwork.h"
#include "activation_fns.h"
#include "training.h"

/* Returns the cost of a neural network given inputs and corresponding outputs */
float cost_fn(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs) {
  Matrix actual_outputs, difference, difference_squared;
  float cost;

  //Calculate the matrix part of the cost function
  actual_outputs = nn_forward(net, inputs);
  difference = matrix_subtract(correct_outputs, &actual_outputs);
  difference_squared = matrix_element_multiply(&difference, &difference); //Element-wise

  //Calculate the scalar part of the cost function
  cost = 0.5 * matrix_sum(&difference_squared);
  
  matrix_free(&actual_outputs);
  matrix_free(&difference);
  matrix_free(&difference_squared);

  return cost;
}

/* Calculates the derivative of the cost function for the given neural network 
 * Note that a[0] = a_2, z[0] = z_2, weights[0] = W1, delta[0] = delta_2, n = total 
 * number of layers (input + hidden + output)
 */
Matrix *cost_fn_prime(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs) {
  int i;
  Matrix actual_outputs, difference, temp_weights, temp_inputs, temp, *z, *a, *dCdW, *delta, **zloc, **aloc;

  /* z = NULL; */
  /* a = NULL; */
  /* zloc = &z; */
  /* aloc = &a; */

  z = malloc(sizeof(Matrix) * (net->numHiddenLayers + 1));
  a = malloc(sizeof(Matrix) * net->numHiddenLayers);
  
  temp_weights = emptyMatrix;
  temp_inputs = emptyMatrix;
  
  //Allocate memory for the output costs (one matrix for each weight matrix)
  dCdW = malloc(sizeof(Matrix) * (net->numHiddenLayers + 1));
  delta = malloc(sizeof(Matrix) * (net->numHiddenLayers + 1)); //Intermediate matrices

  //Calculate y-y_hat
  actual_outputs = nn_forward_activity(net, inputs, z, a);
  difference = matrix_subtract(correct_outputs, &actual_outputs);
  
  matrix_negate(&difference);
  sigmoidPrime_matrix(&z[net->numHiddenLayers]);
  //Equation: delta_n = -(y-yhat) .* sig_prime(z_n) :: element-wise multiplication
  delta[net->numHiddenLayers] = matrix_element_multiply(&difference,
  							&z[net->numHiddenLayers]);
  //DEBUG----------------------------
  /* printf("after assignment %p\n", &delta[net->numHiddenLayers]); */
  /* printf("delta = %p\n", delta); */
  /* //matrix_print(&delta[net->numHiddenLayers], stdout); */
  /* printf("Freeing matrix\n"); */
  /* matrix_free(&delta[net->numHiddenLayers]); */
  /* printf("Freeing delta\n"); */
  /* free(delta); */
  /* printf("Freed delta\n"); */
  /* exit(-1); */
  //End debug------------------------
  
  matrix_transpose(&a[net->numHiddenLayers-1]);
  //Equation: dCdW_n-1 = a_n-1.T * delta_n
  dCdW[net->numHiddenLayers] = matrix_multiply_slow(&a[net->numHiddenLayers-1]                                                    , &delta[net->numHiddenLayers]);
  
  //Apply the chain rule
  for(i = net->numHiddenLayers - 1; i > 0; i--) {
    //Copy the weights to be transposed
    matrix_copy(&temp_weights, &(net->weights[i+1]));
    /* Account for the next layer's bias weights' lack of effect on the output of the current layer
     * Note that this does not need to be done for the n-1'th layer, as there is no bias node going
     *into the output layer */
    if(i < net->numHiddenLayers - 1) {
    	matrix_truncate_row(&temp_weights);
    }
    matrix_transpose(&temp_weights);
    sigmoidPrime_matrix(&z[i]);
    temp = matrix_multiply_slow(&delta[i+1], &temp_weights);
    //Equation: delta_m = (delta_m+1 * Wm.T) .* sig_prime(z_m)
    delta[i] = matrix_element_multiply(&temp, &z[i]);
    matrix_transpose(&a[i-1]);
    //Equation: dCdW_m-1 = a_m-1.T * delta_m
    dCdW[i] = matrix_multiply_slow(&a[i-1], &delta[i]);

    //Free the local varaibles used in each iteration
    matrix_free(&temp_weights);
    matrix_free(&temp);
  }
  
  //Final term (use input matrix)::
  //Copy the weights and inputs to be transposed
  matrix_copy(&temp_weights, &(net->weights[1]));
  matrix_copy(&temp_inputs, inputs);
  append_ones(&temp_inputs);
  if(net->numHiddenLayers != 1) {
    matrix_truncate_row(&temp_weights);
  }
  matrix_transpose(&temp_weights);
  matrix_transpose(&temp_inputs);

  sigmoidPrime_matrix(&z[0]);
  temp = matrix_multiply_slow(&delta[1], &temp_weights);
  //Equation: delta_2 = (delta_3 * W2.T) .* sig_prime(z_2)
  delta[0] = matrix_element_multiply(&temp, &z[0]);
  //Equation: dcdW_1 = inputs.T * delta2
  dCdW[0] = matrix_multiply_slow(&temp_inputs, &delta[0]);

  //Clean up the local variables
  for(i = 0; i < net->numHiddenLayers + 1; i++) {
    matrix_free(&delta[i]);
    matrix_free(&z[i]);
    if(i != net->numHiddenLayers) {
      matrix_free(&a[i]);
    }
  }
  matrix_free(&actual_outputs);
  matrix_free(&difference);
  matrix_free(&temp_weights);
  matrix_free(&temp_inputs);
  matrix_free(&temp);
  
  free(delta);
  free(z);
  free(a);
  return dCdW;
}						     
						     
