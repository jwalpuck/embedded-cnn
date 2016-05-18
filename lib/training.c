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

/* Returns the overall cost via cross entropy (log loss) of a convolutional neural network 
   given a set of inputs and corresponding outputs of size n */
float cross_entropy(Conv_Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, int n) {
  int i, j;
  float cost = 0;
  Matrix cur_outputs;

  //Sum the cost for each input
  for(i = 0; i < n; i++) {
    cur_outputs = cnn_forward(net, &inputs[i]);

    //DEBUG
    printf("Network outputs:\n");
    matrix_print(&cur_outputs, stdout);
    printf("Correct outputs:\n");
    matrix_print(&correct_outputs[i], stdout);

    for(j = 0; j < correct_outputs[i].cols; j++) {
      if(!(fabs(correct_outputs[i].m[0][j]) < 1e-4)) {
	float temp = log(cur_outputs.m[0][j]);
	cost -= isinf(temp) ? -10 : temp; //Big penalty for examples that have 0 for c = y
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
  Matrix **featureMap_yGrads, **pooling_xGrads, **pooling_yGrads;

  temp_weights = emptyMatrix;
  temp_grad = emptyMatrix;

  //printf("Allocating activity matrix arrays\n");
  //Allocate activity matrices
  fullLayerInputs = malloc(sizeof(Matrix) * 2); //One matrix for preactivation, one for activation
  fullLayerInputs[0] = emptyMatrix;
  fullLayerInputs[1] = emptyMatrix;

  // printf("Got the full layer inputs\n");

  //Preactivation for each fully connected hidden layer + output layer
  a = malloc(sizeof(Matrix) * (net->numFullLayers+1));
  //Activation for each fully connected hidden layer
  h = malloc(sizeof(Matrix) * net->numFullLayers);

  y = malloc(sizeof(Matrix *) * net->numHiddenLayers);
  conv_x = malloc(sizeof(Matrix *) * (net->numHiddenLayers+1));
  pool_x = malloc(sizeof(Matrix *) * net->numHiddenLayers);
  max_idxs = malloc(sizeof(Matrix *) * (net->numHiddenLayers+1));

  featureMap_yGrads = malloc(sizeof(Matrix *) * (net->numHiddenLayers+1));
  pooling_xGrads = malloc(sizeof(Matrix *) * net->numHiddenLayers);
  pooling_yGrads = malloc(sizeof(Matrix *) * net->numHiddenLayers);

  //First input to convolution layer is the singular input image
  conv_x[0] = malloc(sizeof(Matrix));
  //max_idxs[0] = malloc(sizeof(Matrix));

  //printf("Allocating subarrays\n");

  for(i = 0; i < net->numHiddenLayers; i++) {
    y[i] = malloc(sizeof(Matrix) * net->depths[i]);
    conv_x[i+1] = malloc(sizeof(Matrix) * net->depths[i]);
    pool_x[i] = malloc(sizeof(Matrix) * net->depths[i]);
    max_idxs[i] = malloc(sizeof(Matrix) * net->depths[i]);
    featureMap_yGrads[i] = malloc(sizeof(Matrix) * net->depths[i]);
    pooling_xGrads[i] = malloc(sizeof(Matrix) * net->depths[i]);
    pooling_yGrads[i] = malloc(sizeof(Matrix) * net->depths[i]);
    for(j = 0; j < net->depths[i]; j++) {
      featureMap_yGrads[i][j] = emptyMatrix;
      pooling_xGrads[i][j] = emptyMatrix;
      pooling_yGrads[i][j] = emptyMatrix;
    }
  }

  //printf("Allocated subarrays\n");

  featureMap_yGrads[net->numHiddenLayers] = malloc(sizeof(Matrix) * net->fullLayerSizes[0]);
  for(i = 0; i < net->fullLayerSizes[0]; i++) {
    featureMap_yGrads[net->numHiddenLayers][i] = emptyMatrix;
  }
  
  //printf("Calculating forward propagation with activity\n");
  actual_outputs = cnn_forward_activity(net, inputs, fullLayerInputs, a, h, y, conv_x, pool_x, max_idxs);
  //See where the outputs are all becoming 0
  int all_zero = 1;
  for(i = 0; i < 10; i++) {
    if(actual_outputs.m[0][i] > 0.1) {
      all_zero = 0;
      break;
    }
  }
  //Print the outputs
  //matrix_print(&actual_outputs, stdout);
  if(all_zero) {
    //matrix_print(&actual_outputs, stdout);
    printf("All zero\n");
    //exit(-1);
  }

  /* printf("Net structure: %d full layers\n", net->numFullLayers); */
  /* for(i = 0; i < net->numFullLayers; i++) { */
  /*   printf("Full layer W[%d] is %dx%d\n", i, net->fullLayerWeights[i].rows, net->fullLayerWeights[i].cols); */
  /* } */

  /*************** Begin back propagation ****************/
  //Calculate output gradient
  //printf("Calculating output gradient\n");
  /* printf("Outputs from forward propagation:\n"); */
  /* matrix_print(&actual_outputs, stdout); */
  /* printf("Label:\n"); */
  /* matrix_print(correct_output, stdout); */
  grad1 = matrix_subtract(correct_output, &actual_outputs);
  matrix_free(&actual_outputs);
  //DEBUG (TEST)
  for(i = 0; i < grad1.cols; i++) {
    if(fabs(correct_output->m[0][i] - 1) > 1e-6) {
      grad1.m[0][i] = 0; //Grad for a of output
    }
  }
  matrix_negate(&grad1);
  
  //printf("Calculated output gradient\n");

  //Gradients for fully connected layers
  for(i = net->numFullLayers; i > 0; i--) {
    /* printf("For layer %d\n", i); */
    /* printf("Current a gradient:\n"); */
    //matrix_print(&grad1, stdout);
    //Calculate gradient of weights
    //matrix_transpose(&h[i-1]);

    //DEBUG
    matrix_transpose(&grad1);
    if(i < net->numFullLayers) {
      //append_ones(&h[i-1]);
    }

    /* printf("h[%d]: \n", i-1); */
    /* matrix_print(&h[i-1], stdout); */

    //printf("Calculating ∇W[%d] with grad1 = %dx%d and h[%d] = %dx%d.\n", i-1, grad1.rows, grad1.cols, i-1, h[i-1].rows, h[i-1].cols);
    FC_gradients[i-1] = matrix_multiply_slow(&grad1, &h[i-1]);
    /* float temp_wMax = matrix_max(&FC_gradients[i-1]); */
    /* if(temp_wMax > 0.5) { */
    /*   printf("This is fishy:\n"); */
    /*   matrix_print(&FC_gradients[i-1], stdout); */
    /*   exit(-1); */
    /* } */
    //matrix_print(&FC_gradients[i-1], stdout);
    //matrix_transpose(&FC_gradients[i-1]);
    
    //DEBUG
    //printf("Calculated FC gradient %d\n", i);
    
    //Calculate gradient of h
    //printf("Copying weights\n");
    
    //Keep in mind that only one FC --> FC and one FC --> output weight matrix exist
    matrix_copy(&temp_weights, &net->fullLayerWeights[i-1]);
    //printf("Copying weights\n");
    //Account for the next layer's bias weights' lack of effect on the current layer output
    //if(i < net->numFullLayers) {
      //printf("Truncating row...\n");
    matrix_truncate_row(&temp_weights);
      //printf("Truncated row.\n");
      //}
    //printf("Transposing weights...\n");
    //matrix_transpose(&temp_weights); //DEBUG
    //printf("Transposed weights\n");
    //printf("Calculating ∇h[%d] with temp_weights = %dx%d and ∇a[%d] = %dx%d\n", i-1, temp_weights.rows, temp_weights.cols, i, grad1.rows, grad1.cols);
    //printf("Temp weights %d:\n", i);
    //matrix_print(&temp_weights, stdout);
    grad2 = matrix_multiply_slow(&temp_weights, &grad1); //Gradient of h[i-1]
    
    //matrix_truncate_row(&grad2);
    matrix_transpose(&grad2); //DEBUG

    /* printf("Gradient of h[%d]:\n", i-1); */
    /* matrix_print(&grad2, stdout); */

    //Calculate gradient of a
    matrix_free(&grad1);
    tanhPrime_matrix(&a[i-1]);
    //printf("Calculating ∇a[%d] with ∇h[%d] = %dx%d and a[%d] = %dx%d\n", i-1, i-1, grad2.rows, grad2.cols, i-1, a[i-1].rows, a[i-1].cols);
    /* printf("g'(a[%d]):\n", i-1); */
    /* matrix_print(&a[i-1], stdout); */
    grad1 = matrix_element_multiply(&grad2, &a[i-1]);

    /* printf("∇a[%d]:\n", i-1); */
    /* matrix_print(&grad2, stdout); */

    //matrix_copy(&grad1, &grad2); //grad1 now holds the gradient of a[i-1]
    matrix_free(&grad2);
    matrix_free(&temp_weights);
  }

  //DEBUG TO CHECK GRADIENTS AFTER FULL LAYER
  /* printf("Mic check:\n"); */
  /* matrix_print(&FC_gradients[0], stdout); */
  /* matrix_print(&FC_gradients[1], stdout); */
  /* printf("*drops mic*\n"); */
  /* exit(-1); */

  //printf("Exited loop 1\n");

  /*****************************************************************************/
  /** TRANSITION FROM FULLY CONNECTED LAYERS TO HIDDEN LAYERS *****************/
  /***************************************************************************/

  //printf("Working on transition...\n");

  //Divide the matrix into an array of 1x1 matrices for easier backpropagation
  matrix_copy(&grad2, &grad1); //For ease of debugging ***
  matrix_free(&grad1);
  //printf("Dividing into %d separate matrices\n", grad2.cols);
  for(i = 0; i < grad2.cols; i++) {
    matrix_init(&featureMap_yGrads[net->numHiddenLayers][i], 1, 1);
    featureMap_yGrads[net->numHiddenLayers][i].m[0][0] = grad2.m[0][i];
    /* printf("Matrix %d:\n", i); */
    /* matrix_print(&featureMap_yGrads[net->numHiddenLayers][i], stdout); */
  }
  matrix_free(&grad2);
  //printf("Divided the matrices\n");

  //Calculate gradient of the kernels going to fully connected layers
  for(i = 0; i < net->fullLayerSizes[0]; i++) {
      //Calculate the gradients of the kernels for the current layer feature maps
    //printf("Outside\n");
    kernel_gradients[net->numHiddenLayers][i] = matrix_tilde_convolution(&conv_x[net->numHiddenLayers][0], &featureMap_yGrads[net->numHiddenLayers][i]);
    //printf("We're going in\n");
    for(j = 1; j < net->depths[net->numHiddenLayers-1]; j++) {
      //printf("%d, %d\n", net->numHiddenLayers, j);
      //matrix_print(&conv_x[net->numHiddenLayers][j], stdout);
      temp_grad = matrix_tilde_convolution(&conv_x[net->numHiddenLayers][j], &featureMap_yGrads[net->numHiddenLayers][i]);
      matrix_add_inPlace(&kernel_gradients[net->numHiddenLayers][i], &temp_grad);
      matrix_free(&temp_grad);
    }
  }

  //printf("Calculated gradient of kernels going into fully connected layers\n");
  //Display transition kernel gradients
  /* for(i = 0; i < net->fullLayerSizes[0]; i++) { */
  /*   printf("Transition kernel %d, %dx%d:\n", i, kernel_gradients[net->numHiddenLayers][i].rows, kernel_gradients[net->numHiddenLayers][i].cols); */
  /*   matrix_print(&kernel_gradients[net->numHiddenLayers][i], stdout); */
  /* } */
 
  //Calculate the gradient of the output of the pooling layer (which was convolved)
  grad1 = matrix_zeroPad_convolution(&featureMap_yGrads[net->numHiddenLayers][0], &net->kernels[net->numHiddenLayers][0]);
  //for(j = 1; j < net->depths[net->numHiddenLayers-1]; j++) {
  for(j = 1; j < net->fullLayerSizes[0]; j++) {
    //Loop over all feature maps. Only need 1 grad for x, use this as output of previous pooling layer
    temp_grad = matrix_zeroPad_convolution(&featureMap_yGrads[net->numHiddenLayers][j], &net->kernels[net->numHiddenLayers][j]);
    matrix_add_inPlace(&grad1, &temp_grad);
    matrix_free(&temp_grad);
  }

  //printf("Calculated gradient of output of the pooling layer\n");
  //matrix_print(&grad1, stdout);

  //To avoid rewriting code (for now)
  matrix_copy(&grad2, &grad1);
  matrix_free(&grad1);
	      
  
    /*******************************************************************************/
    /***************** HIDDEN LAYER GRADIENTS *************************************/
    /*****************************************************************************/

  //Calculate gradients for convolution/pooling layers
  for(i = net->numHiddenLayers-1; i >= 0; i--) {
    //printf("Hidden layer %d:\n", i);
    for(j = 0; j < net->depths[i]; j++) {
      //Calculate gradients of input (x) to pooling layer (activation of previous convolution layer)
      //Reset for each feature map when transitioning to other hidden layers
      count = 0;

      /* printf("Pool_x[%d][%d] has dimensions %dx%d\n", i, j, pool_x[i][j].rows, pool_x[i][j].cols); */
      /* printf("Max_idxs[%d][%d] has %d elements:\n", i, j, max_idxs[i][j].rows * max_idxs[i][j].cols); */
      matrix_init(&grad1, pool_x[i][j].rows, pool_x[i][j].cols);
      //printf("Grad1 initialized: %dx%d\n", grad1.rows, grad1.cols);
      for(k = 0; k < max_idxs[i][j].rows; k++) {
	//printf("Entered reverse pooling loop\n");
	row_idx = max_idxs[i][j].m[k][0];
	col_idx = max_idxs[i][j].m[k][1];
	
	//printf("Iteration %d, assigning to %d, %d from %d, %d\n", k, row_idx, col_idx, count/grad2.cols, count%grad2.cols);
	grad1.m[row_idx][col_idx] = grad2.m[count/grad2.cols][count%grad2.cols];
	//printf("Added number: %f from indices %d, %d\n", grad1.m[row_idx][col_idx], row_idx, col_idx);
	count++;
      }
      matrix_copy(&pooling_xGrads[i][j], &grad1);
      if(i == net->numHiddenLayers-1) {
	/* printf("pooling_xGrad[%d][%d]:\n", i, j); */
	/* matrix_print(&grad1, stdout); */
      }
      matrix_free(&grad1);
    }

    /* printf("Grad2a\n"); */
    /* matrix_print(&grad2, stdout); */
    /* printf("end grad2a\n"); */
    matrix_free(&grad2);

    //Calculate gradients of previous convolution layer y (preactivation)
    for(j = 0; j < net->depths[i]; j++) {

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
      
      /* printf("Grad2b\n"); */
      /* matrix_print(&grad2, stdout); */
      /* printf("end grad2b\n"); */
      matrix_free(&grad2);
    }

    for(j = 0; j < net->depths[i]; j++) {
      //Calculate the gradients of the kernels for the current layer feature maps
      kernel_gradients[i][j] = matrix_tilde_convolution(&conv_x[i][0], &featureMap_yGrads[i][j]);
      if(i != 0) { //If more than one input (not original inputs), must consider them all
	for(k = 1; k < net->depths[i-1]; k++) {
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
  matrix_free(&grad2);

  /* printf("Kernel Gradient[2][0]:\n"); */
  /* matrix_print(&kernel_gradients[2][0], stdout); */

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
      matrix_free(&pooling_yGrads[i][j]);
    }
    free(y[i]);
    free(conv_x[i+1]);
    free(pool_x[i]);
    free(max_idxs[i]);
    free(featureMap_yGrads[i]);
    free(pooling_xGrads[i]);
    free(pooling_yGrads[i]);
  }

  for(i = 0; i < net->fullLayerSizes[0]; i++) {
    matrix_free(&featureMap_yGrads[net->numHiddenLayers][i]);
  }
  free(featureMap_yGrads[net->numHiddenLayers]);

  free(conv_x[0]);
  free(y);
  free(conv_x);
  free(pool_x);
  free(max_idxs);
  free(featureMap_yGrads);
  free(pooling_xGrads);
  free(pooling_yGrads);
  

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
  k_gradients = malloc(sizeof(Matrix *) * (net->numHiddenLayers + 1));
  fc_gradients = malloc(sizeof(Matrix) * (net->numFullLayers));
  for(i = 0; i < net->numHiddenLayers; i++) {
    k_gradients[i] = malloc(sizeof(Matrix) * net->depths[i]);
  }
  k_gradients[net->numHiddenLayers] = malloc(sizeof(Matrix) * net->fullLayerSizes[0]);

  for(count = 0; count < n; count++) {
    for(i = 0; i < numInputs; i++) {		
      //printf("Training example %d\n", i);
      //Calculate the gradients for the i'th input
      cnn_backProp(net, &inputs[i], &correct_outputs[i], k_gradients, fc_gradients);

      /* printf("Kernel Gradient[2][0]:\n"); */
      /* matrix_print(&k_gradients[2][0], stdout); */
      //exit(-1);
      
      //Update the weights of the neural network
      //printf("Updating weights...\n");
      cnn_updateWeights(net, k_gradients, fc_gradients, learningRate);
      //printf("Updated weights\n");

      /* printf("Kernel gradients:\n"); */
      /* for(j = 0; j < net->numHiddenLayers; j++) { */
      /* 	for(k = 0; k < net->depths[j]; k++) { */
      /* 	  matrix_print(&k_gradients[j][k], stdout); */
      /* 	} */
      /* } */

      if((i+1) % 20000 == 0) {
      /* 	/\* printf("At iteration %d\n", i); *\/ */
      /* 	/\* printf("\nBackpropagated input %d\n", i+1); *\/ */
      /* 	/\* printf("Kernel gradients:\n"); *\/ */
      /* 	/\* for(j = 0; j < net->numHiddenLayers; j++) { *\/ */
      /* 	/\*   for(k = 0; k < net->depths[j]; k++) { *\/ */
      /* 	/\*     matrix_print(&k_gradients[j][k], stdout); *\/ */
      /* 	/\*   } *\/ */
      /* 	/\* } *\/ */
      /* 	printf("Fully connected layer weight gradients\n"); */
      /* 	for(j = 0; j < net->numFullLayers; j++) { */
      /* 	  matrix_print(&fc_gradients[j], stdout); */
      /* 	} */
      	return;
       }
		
      //printf("Ahh, okay\n");
      //Free gradients to prevent memory leaks on subsequent iterations
      for(j = 0; j < net->numFullLayers; j++) {
      	matrix_free(&fc_gradients[j]);
      }
      for(j = 0; j < net->numHiddenLayers; j++) {
	for(k = 0; k < net->depths[j]; k++) {
	  matrix_free(&k_gradients[j][k]);
	}
      }
      for(j = 0; j < net->fullLayerSizes[0]; j++) {
	matrix_free(&k_gradients[net->numHiddenLayers][j]);
      }
    }
    
    //Can add this later
    /* if(n > 1) { */
    /*   //Shuffle the rows of the training data before doing another gradient descent pass */
    /*   matrix_shuffle_rows(inputs, correct_outputs); */
    /* } */
  }
  
  //Clean up
  for(i = 0; i < net->numHiddenLayers+1; i++) {
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

void cnn_gradientChecker(Conv_Neural_Network *net, Matrix *input, Matrix *output) {
  int i;
  Matrix **k_gradients, *fc_gradients, **k_approx, *fc_approx;

  //Initialize arrays for the gradients to be stored in
  k_gradients = malloc(sizeof(Matrix *) * (net->numHiddenLayers + 1));
  fc_gradients = malloc(sizeof(Matrix) * (net->numFullLayers+1));
  for(i = 0; i < net->numHiddenLayers; i++) {
    k_gradients[i] = malloc(sizeof(Matrix) * net->depths[i]);
  }
  k_gradients[net->numHiddenLayers] = malloc(sizeof(Matrix) * net->fullLayerSizes[0]);

  //Get gradient values from back propagation algorithm
  cnn_backProp(net, input, output, k_gradients, fc_gradients);

  //Calculate approximate gradients
  
}
