/**
 * training.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "neuralNetwork.h"
#include "conv_neuralNetwork.h"
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

/* Returns the overall cost via cross entropy of a convolutional neural network given
   a set of inputs and corresponding outputs of size n */
float cross_entropy(Conv_Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, int n) {
  int i, j;
  float cost = 0;
  Matrix cur_outputs;

  //Sum the cost for each input
  for(i = 0; i < n; i++) {
    cur_outputs = cnn_forward(net, &inputs[i]);
    for(j = 0; j < correct_outputs[i].cols; j++) {
      if(fabs(correct_outputs[i].m[0][j]) - 1 < 1e-6) {
	cost -= log(cur_outputs.m[0][j]);
	break;
      }
    }
    matrix_free(&cur_outputs);
  }

  return cost;
}

/* Calculates the derivative of the cost function for the given neural network 
 * Note that a[0] = a_2, z[0] = z_2, weights[0] = W1, delta[0] = delta_2, n = total 
 * number of layers (input + hidden + output)
 */
Matrix *cost_fn_prime(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs) {
  int i;
  Matrix actual_outputs, difference, temp_weights, temp_inputs, temp, *z, *a, *dCdW, *delta;

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

/* Calculates the gradients for all kernels and weight matrices in net given a set of 
   inputs, their correct outputs, and a CNN to run them through */
void cnn_backProp(Conv_Neural_Network *net, Matrix *inputs, Matrix *correct_output, Matrix **kernel_gradients, Matrix *FC_gradients) {
  int i, j, k, row_idx, col_idx, count;
  Matrix *fullLayerInputs, *a, *h, **y, **conv_x, **pool_x, **max_idxs;
  Matrix actual_outputs, temp_weights, grad1, grad2, temp_grad;
  Matrix **featureMap_yGrads, **pooling_xGrads;

  temp_weights = emptyMatrix;
  temp_grad = emptyMatrix;

  //Allocate activity matrices
  fullLayerInputs = malloc(sizeof(Matrix));

  //Preactivation for each fully connected hidden layer + output layer
  a = malloc(sizeof(Matrix) * (net->numFullLayers+1));
  //Activation for each fully connected hidden layer
  h = malloc(sizeof(Matrix) * net->numFullLayers);

  y = malloc(sizeof(Matrix *) * net->numHiddenLayers);
  conv_x = malloc(sizeof(Matrix *) * (net->numHiddenLayers+1));
  pool_x = malloc(sizeof(Matrix *) * net->numHiddenLayers);
  max_idxs = malloc(sizeof(Matrix *) * net->numHiddenLayers);

  featureMap_yGrads = malloc(sizeof(Matrix *) * net->numHiddenLayers);
  pooling_xGrads = malloc(sizeof(Matrix *) * net->numHiddenLayers);

  //First input to convolution layer is the singular input image
  conv_x[0] = malloc(sizeof(Matrix));

  for(i = 0; i < net->numHiddenLayers; i++) {
    y[i] = malloc(sizeof(Matrix) * net->depths[i]);
    conv_x[i+1] = malloc(sizeof(Matrix) * net->depths[i]);
    pool_x[i] = malloc(sizeof(Matrix) * net->depths[i]);
    max_idxs[i] = malloc(sizeof(Matrix) * net->depths[i]);
    featureMap_yGrads[i] = malloc(sizeof(Matrix) * net->depths[i]);
    pooling_xGrads[i] = malloc(sizeof(Matrix) * net->depths[i]);
    for(j = 0; j < net->depths[i]; j++) {
      featureMap_yGrads[i][j] = emptyMatrix;
      pooling_xGrads[i][j] = emptyMatrix;
    }
  }
  
  //printf("Calculating forward propagation with activity\n");

  actual_outputs = cnn_forward_activity(net, inputs, fullLayerInputs, a, h, y, conv_x, pool_x, max_idxs);

  /*************** Begin back propagation ****************/
  //Calculate output gradient
  //printf("Calculating output gradient\n");
  grad1 = matrix_subtract(correct_output, &actual_outputs);
  //DEBUG (TEST)
  for(i = 0; i < grad1.cols; i++) {
    if(fabs(correct_output->m[0][i] - 1) > 1e-6) {
      grad1.m[0][i] = 0;
    }
  }
  matrix_negate(&grad1);
  
  //printf("Calculated output gradient\n");

  //Gradients for fully connected layers
  for(i = net->numFullLayers; i > 0; i--) {
    //printf("For layer %d\n", i);
    //Calculate gradient of weights
    //matrix_transpose(&h[i-1]);

    //DEBUG
    matrix_transpose(&grad1);
    if(i < net->numFullLayers) {
      append_ones(&h[i-1]);
    }

    FC_gradients[i] = matrix_multiply_slow(&grad1, &h[i-1]);
    
    //DEBUG
    //printf("Calculated FC gradient %d\n", i);
    
    //Calculate gradient of h
    //printf("Copying weights\n");
    matrix_copy(&temp_weights, &net->fullLayerWeights[i]);
    //printf("Copying weights\n");
    //Account for the next layer's bias weights' lack of effect on the current layer output
    if(i < net->numFullLayers) {
      //printf("Truncating row...\n");
      matrix_truncate_row(&temp_weights);
      //printf("Truncated row.\n");
    }
    //printf("Transposing weights...\n");
    //matrix_transpose(&temp_weights); //DEBUG
    //printf("Transposed weights\n");
    grad2 = matrix_multiply_slow(&temp_weights, &grad1);
    matrix_transpose(&grad2); //DEBUG

    //Calculate gradient of a
    matrix_free(&grad1);
    tanhPrime_matrix(&a[i-1]);
    matrix_element_multiply(&grad2, &a[i-1]);

    matrix_copy(&grad1, &grad2);
    matrix_free(&grad2);
    matrix_free(&temp_weights);
  }

  //printf("Exited loop 1\n");

  //Transition from convolution/pooling layers to fully connected layers
  append_ones(fullLayerInputs); //To account for operation done for biases
  float fl_max1 = matrix_max(fullLayerInputs);
  float fl_max2 = matrix_max(&grad1);
  if(fabs(fl_max1) > 2 || fabs(fl_max2) > 5) {
    printf("Maxes from full layer gradient: %f, %f\n", fl_max1, fl_max2);
    exit(-1);
  }
  //matrix_transpose(fullLayerInputs); //DEBUG
  matrix_transpose(&grad1); //DEBUG
  FC_gradients[0] = matrix_multiply_slow(&grad1, fullLayerInputs);

  float gr_max = matrix_max(&FC_gradients[0]);
  if(fabs(gr_max) > 100) {
    printf("Updated gradient to have val %f\n", gr_max);
  }
  
  //Gradient of y for the last pooling layer
  matrix_copy(&temp_weights, &net->fullLayerWeights[0]);
  matrix_truncate_row(&temp_weights);
  grad2 = matrix_multiply_slow(&temp_weights, &grad1);

  //Calculate gradients for convolution/pooling layers
  for(i = net->numHiddenLayers-1; i >= 0; i--) {
    count = 0;
    for(j = 0; j < net->depths[i]; j++) {
      //Calculate gradients of input (x) to pooling layer (activation of previous convolution layer)
      //Reset for each feature map when transitioning to other hidden layers
      if(i < net->numHiddenLayers-1) {
	count = 0;
      }
      matrix_init(&grad1, pool_x[i][j].rows, pool_x[i][j].cols);
      for(k = 0; k < max_idxs[i][j].rows; k++) {
	row_idx = max_idxs[i][j].m[k][0];
	col_idx = max_idxs[i][j].m[k][1];
	grad1.m[row_idx][col_idx] = grad2.m[count/grad2.cols][count%grad2.cols];
	count++;
      }
      matrix_copy(&pooling_xGrads[i][j], &grad1);
      //matrix_print(&grad1, stdout);
      matrix_free(&grad1);
    }

    /* printf("Grad2a\n"); */
    /* matrix_print(&grad2, stdout); */
    /* printf("end grad2a\n"); */
    matrix_free(&grad2);

    //Calculate gradients of previous convolution layer y (preactivation)
    for(j = 0; j < net->depths[i]; j++) {



      //sigmoid_matrix(&y[i][j]); //TEST (IS THIS A GOOD IDEA?)





      tanhPrime_matrix(&y[i][j]);
      float debug_max = matrix_max(&y[i][j]);
      if(debug_max > 1 || debug_max < 0) {
	printf("y[%d][%d] max = %f\n", i, j, matrix_max(&y[i][j]));
      }
      /* printf("Y[%d][%d]\n", i, j); */
      /* matrix_print(&y[i][j], stdout); */
      /* printf("end y\n"); */
      grad2 = matrix_element_multiply(&pooling_xGrads[i][j], &y[i][j]);
      matrix_copy(&featureMap_yGrads[i][j], &grad2);
    }

    /* printf("Grad2b\n"); */
    /* matrix_print(&grad2, stdout); */
    /* printf("end grad2b\n"); */
    matrix_free(&grad2);

    for(j = 0; j < net->depths[i]; j++) {
      //Calculate the gradients of the kernels for the current layer feature maps
      //kernel_gradients[i][j] = matrix_tilde_convolution(&featureMap_yGrads[i][j], &conv_x[i][0]);
      kernel_gradients[i][j] = matrix_tilde_convolution(&conv_x[i][0], &featureMap_yGrads[i][j]);
      if(i != 0) { //If more than one input (not original inputs), must consider them all
	for(k = 1; k < net->depths[i-1]; k++) {
	  //temp_grad = matrix_tilde_convolution(&featureMap_yGrads[i][j], &conv_x[i][k]);
	  temp_grad = matrix_tilde_convolution(&conv_x[i][k], &featureMap_yGrads[i][j]);
	  matrix_add_inPlace(&kernel_gradients[i][j], &temp_grad);
	  matrix_free(&temp_grad);
	}
      }
    }

    /* printf("Grad2c\n"); */
    /* matrix_print(&grad2, stdout); */
    /* printf("end grad2c\n") */;
    matrix_free(&grad1);

    //Calculate the gradient of the input (x) to the current layer
    grad1 = matrix_zeroPad_convolution(&featureMap_yGrads[i][0], &net->kernels[i][0]);
    for(j = 1; j < net->depths[i]; j++) {
      //Loop over all feature maps. Only need 1 grad 1 for x, use this as output of previous pooling layer
      temp_grad = matrix_zeroPad_convolution(&featureMap_yGrads[i][j], &net->kernels[i][j]);
      matrix_add_inPlace(&grad1, &temp_grad);
      matrix_free(&temp_grad);
    }
    matrix_copy(&grad2, &grad1); //For smooth transition back to the top of the loop
    /* printf("Grad2d\n"); */
    /* matrix_print(&grad2, stdout); */
    /* printf("end grad2d\n"); */
    matrix_free(&grad1);
  }

  //Clean up
  //printf("Cleaning up!\n");
  matrix_free(fullLayerInputs);
  //printf("Freeing standard address\n");
  free(fullLayerInputs);

  //printf("Freed full layer inputs\n");
  
  for(i = 0; i < net->numFullLayers+1; i++) {
    matrix_free(&a[i]);
    if(i < net->numFullLayers) {
      matrix_free(&h[i]);
    }
  }
  free(a);
  free(h);

  matrix_free(&conv_x[0][0]);
  for(i = 0; i < net->numHiddenLayers; i++) {
    for(j = 0; j < net->depths[i]; j++) {
      matrix_free(&y[i][j]);
      matrix_free(&conv_x[i+1][j]);
      matrix_free(&pool_x[i][j]);
      matrix_free(&max_idxs[i][j]);
      matrix_free(&featureMap_yGrads[i][j]);
      matrix_free(&pooling_xGrads[i][j]);
    }
    free(y[i]);
    free(conv_x[i+1]);
    free(pool_x[i]);
    free(max_idxs[i]);
    free(featureMap_yGrads[i]);
    free(pooling_xGrads[i]);
  }
  free(conv_x[0]);
  free(y);
  free(conv_x);
  free(pool_x);
  free(max_idxs);
  free(featureMap_yGrads);
  free(pooling_xGrads);
  

  //printf("Returning!\n");
}		
		     
/* Calculates a new set of weights for the given Neural Network using the stochastic gradient
 * descent optimization algorithm. The network is trained on the parameterized matrices of inputs
 * and their corresponding correct outputs. The gradient will be recalculated n * numInputs times */						     
void stochastic_grad_descent(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, int n, float learningRate) {
  int i, j, count;
  Matrix inputRow, outputRow, *gradients;

  for(count = 0; count < n; count++) {
    for(i = 0; i < inputs->rows; i++) {
      //Get the i'th row from the input and output matrices
      inputRow = get_nth_row(inputs, i);
      outputRow = get_nth_row(correct_outputs, i);
			
      //Calculate the gradient for the i'th row
      gradients = cost_fn_prime(net, &inputRow, &outputRow);
      			
      //Update the weights of the neural network
      nn_updateWeights(net, gradients, learningRate);
			
      matrix_free(&inputRow);
      matrix_free(&outputRow);
      for(j = 0; j < net->numHiddenLayers+1; j++) {
      	matrix_free(&gradients[j]); //Prevent memory leak
      }
      free(gradients);
    }
		
    if(n > 1) {
      //Shuffle the rows of the training data before doing another gradient descent pass
      matrix_shuffle_rows(inputs, correct_outputs);
    }
  }
}

/* Calculates a new set of weights for the given CNN using back propagation with the given
   expected inputs and outputs. The gradient will be recalculated n * numInputs times.
   The parameterized inputs and outputs should be arrays of numInputs matrices. */
void cnn_stochastic_grad_descent(Conv_Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, int n, int numInputs, float learningRate) {
  int i, j, k, count;
  Matrix **k_gradients, *fc_gradients;

  //Initialize arrays for the gradients to be stored in
  k_gradients = malloc(sizeof(Matrix *) * net->numHiddenLayers);
  fc_gradients = malloc(sizeof(Matrix) * (net->numFullLayers+1));
  for(i = 0; i < net->numHiddenLayers; i++) {
    k_gradients[i] = malloc(sizeof(Matrix) * net->depths[i]);
  }

  for(count = 0; count < n; count++) {
    for(i = 0; i < numInputs; i++) {		
      printf("Training example %d\n", i);
      //Calculate the gradients for the i'th input
      cnn_backProp(net, &inputs[i], &correct_outputs[i], k_gradients, fc_gradients);
      
      //Update the weights of the neural network
      cnn_updateWeights(net, k_gradients, fc_gradients, learningRate);

      if((i) % 5000 == 0) {
      	printf("\nBackpropagated input %d\n", i+1);
      	printf("Kernel gradients:\n");
      	for(j = 0; j < net->numHiddenLayers; j++) {
      	  for(k = 0; k < net->depths[j]; k++) {
      	    matrix_print(&k_gradients[j][k], stdout);
      	  }
      	}
      	printf("Fully connected layer weight gradients\n");
      	for(j = 0; j < net->numFullLayers; j++) {
      	  matrix_print(&fc_gradients[j], stdout);
      	}
      	//return;
      }
			
      //Free gradients to prevent memory leaks on subsequent iterations
      for(j = 0; j < net->numFullLayers+1; j++) {
      	matrix_free(&fc_gradients[j]);
      }
      for(j = 0; j < net->numHiddenLayers; j++) {
	for(k = 0; k < net->depths[j]; k++) {
	  matrix_free(&k_gradients[j][k]);
	}
      }
    }
    
    //Can add this later
    /* if(n > 1) { */
    /*   //Shuffle the rows of the training data before doing another gradient descent pass */
    /*   matrix_shuffle_rows(inputs, correct_outputs); */
    /* } */
  }
  
  //Clean up
  for(i = 0; i < net->numHiddenLayers; i++) {
    free(k_gradients[i]);
  }
  free(k_gradients);
  free(fc_gradients);
}

/* Calculates a new set of weights for the given Neural Network using the stochastic gradient
 * descent optimization algorithm with momentum. The network is trained on the parameterized matrices 
 * of inputs and their corresponding correct outputs. The gradient will be recalculated n * numInputs times */ 
void stochastic_grad_descent_momentum(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, 
				      int n, float learningRate, float momentumRatio) {
  int i, j, count, gradSize;
  Matrix inputRow, outputRow, *curGrad, *prevGrad;
  gradSize = net->numHiddenLayers+1;
  prevGrad = NULL;

  for(count = 0; count < n; count++) {
    for(i = 0; i < inputs->rows; i++) {
      //Get the i'th row from the input and output matrices
      inputRow = get_nth_row(inputs, i);
      outputRow = get_nth_row(correct_outputs, i);
			
      //Calculate the gradient for the i'th row
      curGrad = cost_fn_prime(net, &inputRow, &outputRow);
			
      //Perform the momentum calculation
      if(i > 0) {
	matrix_momentum(curGrad, prevGrad, gradSize, momentumRatio);
      }
			
      //Update the weights of the neural network
      nn_updateWeights(net, curGrad, learningRate);
			
      matrix_free(&inputRow);
      matrix_free(&outputRow);
			
      if(i > 0) { //There is no prevGrad on the first iteration
	for(j = 0; j < gradSize; j++) {
	  matrix_free(&prevGrad[j]); //Prevent memory leak
	}
	free(prevGrad);
      }
      prevGrad = curGrad; //Prepare momentum matrix for the next iteration
    }
		
    if(n > 1) {
      //Shuffle the rows of the training data before doing another gradient descent pass
      matrix_shuffle_rows(inputs, correct_outputs);
    }
  }
  
}
