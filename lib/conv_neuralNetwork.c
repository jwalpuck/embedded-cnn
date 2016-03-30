/**
  * conv_neuralNetwork.c
  * Jack Walpuck
  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix.h"
#include "conv_neuralNetwork.h"
#include "activation_fns.h"

/* Initialize a convolutional neural network with the given structure */
void cnn_init(Conv_Neural_Network *net, int numOutputs, int numHiddenLayers,
	      int numFullLayers, int *fullLayerSizes, int *kernelSizes, int *depths, int stride) {
  net->outputLayerSize = numOutputs;
  net->numHiddenLayers = numHiddenLayers;
  net->numFullLayers = numFullLayers;
  net->fullLayerSizes = fullLayerSizes;
  net->kernelSizes = kernelSizes;
  net->depths = depths;
  net->stride = stride;

  /* Need one kernel for each convolution layer */
  net->kernels = malloc(sizeof(Matrix *) * net->numHiddenLayers);
  
  /* Need one weight matrix for each fully connected layer plus the output layer */
  net->fullLayerWeights = malloc(sizeof(Matrix) * (net->numFullLayers + 1));
  cnn_generateWeights(net);
}

/* Free the dynamically allocated memory in the convolutional neural network */
void cnn_free(Conv_Neural_Network *net) {
  if(net) {
    int i;
    if(net->kernels) {
      int j;
      for(i = 0; i < net->numHiddenLayers; i++) {
	for(j = 0; j < net->depths[i]; j++) {
	  matrix_free(&net->kernels[i][j]);
	}
	free(net->kernels[i]);
      }
      free(net->kernels);
    }
    if(net->fullLayerWeights) {
      for(i = 0; i < net->numFullLayers; i++) {
	matrix_free(&net->fullLayerWeights[i]);
      }	
      free(net->fullLayerWeights);
    }
  }
}

/* Initialize a convolutional neural network with a pre-existing set of weights */
void cnn_initWithWeights(Conv_Neural_Network *net, int numOutputs, int numHiddenLayers, 
			 int numFullLayers, int *fullLayerSizes, int *kernelSizes, 
			 int *depths, int stride, Matrix *weights) {
			
}

/* Initialize the weight vectors with random floats */
void cnn_generateWeights(Conv_Neural_Network *net){
  int i, j, k, d;
  float randWeight, upperBound = 0.5;

  //srand(time(NULL)); //COMMENT OUT FOR SAME SET OF RANDOM NUMBERS EACH TIME
  srand(1);
  
  //Need one weight/kernel matrix for each hidden layer
  for(i = 0; i < net->numHiddenLayers; i++) {
    net->kernels[i] = malloc(sizeof(Matrix) * net->depths[i]);
    for(d = 0; d < net->depths[i]; d++) {
      matrix_init(&(net->kernels[i][d]), net->kernelSizes[i], net->kernelSizes[i]);
      for(j = 0; j < net->kernels[i][d].rows; j++) { //Populate the kernels with random weights
	for(k = 0; k < net->kernels[i][d].cols; k++) {
	  //Generate a random weight in [-upperBound, upperBound]
	  randWeight = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
	  net->kernels[i][d].m[j][k] = randWeight;
	}
      }
    }
  }
	
  /********* Assign weights to fully connected/output layers *********/

  if(net->numFullLayers == 0) { //Go straight to output weights and return
    matrix_init(&(net->fullLayerWeights[0]), net->depths[net->numHiddenLayers-1], net->fullLayerSizes[0]);
    for(j = 0; j < net->fullLayerWeights[0].rows; j++) { //Populate the weight matrices with random weights
      for(k = 0; k < net->fullLayerWeights[0].cols; k++) {
	//Generate a random weight in [-upperBound, upperBound]
      	randWeight = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
      	net->fullLayerWeights[i].m[j][k] = randWeight;
      }
    }
    return;
  }
  else { //Initialize weight matrix going into first complete layer (create an extra row for biases)
    matrix_init(&(net->fullLayerWeights[0]), net->depths[net->numHiddenLayers-1]+1, net->fullLayerSizes[0]);
  }
  for(j = 0; j < net->fullLayerWeights[0].rows; j++) { //Populate the kernels with random weights
    for(k = 0; k < net->fullLayerWeights[0].cols; k++) {
      //Generate a random weight in [-upperBound, upperBound]
      randWeight = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
      net->fullLayerWeights[0].m[j][k] = randWeight;
    }
  }
	
  //Need one weight matrix for each fully connected layer
  for(i = 1; i < net->numFullLayers; i++) {
    matrix_init(&(net->fullLayerWeights[i]), net->fullLayerSizes[i-1]+1, net->fullLayerSizes[i]);
    for(j = 0; j < net->fullLayerWeights[i].rows; j++) { //Populate the weight matrices with random weights
      for(k = 0; k < net->fullLayerWeights[i].cols; k++) {
	//Generate a random weight in [-upperBound, upperBound]
      	randWeight = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
      	net->fullLayerWeights[i].m[j][k] = randWeight;
      }
    }
  }

  int finalIndex = net->numFullLayers-1; //For ease of access
  matrix_init(&(net->fullLayerWeights[finalIndex+1]), net->fullLayerSizes[finalIndex], net->outputLayerSize);
  for(j = 0; j < net->fullLayerSizes[finalIndex]; j++) {
    for(k = 0; k < net->outputLayerSize; k++) {
      //Generate a random weight in [-upperBound, upperBound]
      randWeight = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
      net->fullLayerWeights[finalIndex+1].m[j][k] = randWeight;
    }
  }
	
}

/* Forward propogate inputs through the network and return the output vector */
Matrix cnn_forward(Conv_Neural_Network *net, Matrix *inputs){ 
  Matrix *curFeatureMaps, *prevFeatureMaps, temp1, temp2, metro1, metro2;
  int i, j, d, curDepth, prevDepth;

  //Assign default values to make the compiler happy
  curFeatureMaps = NULL;
  curDepth = 0;
	
  //Do depth[i-1] iterations over feature maps for convolutions, then sum all pre-activations before pooling
  //At i=0, only do one iteration (input is 1 dimensional)
	

  //Treat inputs as previous feature maps
  prevFeatureMaps = inputs;
  prevDepth = 1;
  
  //Alternate convolution and pooling operations for numHiddenLayers iterations
  for(i = 0; i < net->numHiddenLayers; i++) {
    curDepth = net->depths[i];
    curFeatureMaps = malloc(sizeof(Matrix) * curDepth);
		
    //Perform a convolution for each feature map over input channels
    for(j = 0; j < curDepth; j++) {
      //Get the first feature map component to add the others to
      temp1 = matrix_convolution(&prevFeatureMaps[0], &net->kernels[i][j]);
      //printf("%dth input channel size: %dx%d\n", 0, temp1.rows, temp1.cols);
      for(d = 1; d < prevDepth; d++) { 
	//Add the convolution results from other feature maps from the previous layer
	temp2 = matrix_convolution(&prevFeatureMaps[d], &net->kernels[i][j]);
	matrix_add_inPlace(&temp1, &temp2);
	matrix_free(&temp2);
      }
      /* Assign the feature map matrix now that all input from the 
	 previous layer has been summed */
      curFeatureMaps[j] = temp1; 
    }
		
    //Pool the result of the convolution with 2x2 non-overlapping neighborhoods
    for(j = 0; j < curDepth; j++) {
      //Apply pointwise non-linearity before ReLU activation? (Yann LeCunn et al, 1998)
      tanh_matrix(&(curFeatureMaps[j]));
      if(i < net->numHiddenLayers-1);
      matrix_pool(&(curFeatureMaps[j]), 2);
    }


    if(i != 0) { //Free matrices from the previous layer
      for(j = 0; j < net->depths[i-1]; j++) {
	matrix_free(&prevFeatureMaps[j]);
      }
    }

    //Store the output of the current layer for operations from the next layer
    prevFeatureMaps = curFeatureMaps;
    prevDepth = curDepth;
    curFeatureMaps = NULL;

  }
	
  //Standard forward propagation from multi-layer perceptron network for fully connected layers

  //printf("Beginning MLP-style forward propagation\n");

  /* First, make sure we only have one value from each feature map being passed into the
     fully connected layers */
  matrix_arrayToMaxMat(&metro1, prevFeatureMaps, curDepth);

  for(i = 0; i < net->numFullLayers + 1; i++) {
    if(i != net->numFullLayers) { //Account for biases in all but the output layer
      append_ones(&metro1);
    }
    metro2 = matrix_multiply_slow(&metro1, &(net->fullLayerWeights[i]));
    if(i != net->numFullLayers) {
      sigmoid_matrix(&metro2);
    }
    else {
      softmax_matrix(&metro2);
    }
    matrix_copy(&metro1, &metro2);
    matrix_free(&metro2);
  }

  //Clean up
  for(i = 0; i < net->depths[net->numHiddenLayers-1]; i++) { //Matrices from the last pooling layer
    matrix_free(&prevFeatureMaps[i]);
  }
	
  return metro1;
}

/* Forward propogate inputs through the network, return ouput and activity matrices
   from each layer (by reference)
   
   @params -- Existing structs
   net: Convolutional Neural Network with pre-initialized weights
   inputs: Matrix containing the input image for forward propagation

   @params -- Empty memory blocks to be filled with information for back propagation
   fullLayerInputs: To store the output of the final pooling layer
   a: To store the pre-activation of fully connected layers
   h: To store the output of fully connected layers
   y: To store the output of convolution layers
   conv_x: To store the input to convolution layers (output of 0:n-1 pooling layers)
   pool_x: To store the input to pooling layers (tanh(y))
   max_idxs: To store the indices of the inputs to pooling layers that were maximum in 
             their pooling neighborhoods
*/
Matrix cnn_forward_activity(Conv_Neural_Network *net, Matrix *inputs, Matrix *fullLayerInputs, Matrix *a, Matrix *h, Matrix **y, Matrix **conv_x, Matrix **pool_x, Matrix **max_idxs) {
  Matrix *curFeatureMaps, *prevFeatureMaps, temp1, temp2, metro1, metro2;
  int i, j, d, curDepth, prevDepth;

  //Assign default values to make the compiler happy
  curFeatureMaps = NULL;
  curDepth = 0;

  //Assign empty matrices
  *fullLayerInputs = emptyMatrix;

  for(i = 0; i < net->numFullLayers+1; i++) {
    a[i] = emptyMatrix;
    if(i != net->numFullLayers) { //Do not need to store the output
      h[i] = emptyMatrix;
    }
  }

  conv_x[0][0] = emptyMatrix;
  for(i = 0; i < net->numHiddenLayers; i++) {
    for(j = 0; j < net->depths[i]; j++) {
      y[i][j] = emptyMatrix;
      conv_x[i+1][j] = emptyMatrix;
      pool_x[i][j] = emptyMatrix;
      max_idxs[i][j] = emptyMatrix;
    }
  }

  //Perform forward propagation, storing the necessary matrices in the allocated blocks
	
  //Do depth[i-1] iterations over feature maps for convolutions, then sum all pre-activations before pooling
  //At i=0, only do one iteration (input is 1 dimensional)
	

  //Treat inputs as previous feature maps
  prevFeatureMaps = inputs;
  matrix_copy(&conv_x[0][0], inputs);
  prevDepth = 1;
  
  //Alternate convolution and pooling operations for numHiddenLayers iterations
  for(i = 0; i < net->numHiddenLayers; i++) {
    //printf("Hidden layer %d\n", i);
    curDepth = net->depths[i];
    curFeatureMaps = malloc(sizeof(Matrix) * curDepth);
    for(j = 0; j < curDepth; j++) {
      curFeatureMaps[j] = emptyMatrix;
    }
		
    //Perform a convolution for each feature map over input channels
    //printf("Convolving\n");
    for(j = 0; j < curDepth; j++) {
      //Get the first feature map component to add the others to
      temp1 = matrix_convolution(&prevFeatureMaps[0], &net->kernels[i][j]);

      for(d = 1; d < prevDepth; d++) { 
	//Add the convolution results from other feature maps from the previous layer
	temp2 = matrix_convolution(&prevFeatureMaps[d], &net->kernels[i][j]);
	matrix_add_inPlace(&temp1, &temp2);
	matrix_free(&temp2);
      }

      //Store pre-activation convolution output for backpropagation
      matrix_copy(&y[i][j], &temp1);

      /* Apply nonlinearity and assign the feature map matrix now that all input from the 
	 previous layer has been summed */
      tanh_matrix(&temp1);
      matrix_copy(&curFeatureMaps[j], &temp1);

      //Store convolution output (activated) for backpropagation
      matrix_copy(&pool_x[i][j], &temp1);
      matrix_free(&temp1); //DEBUG
    }
    
    //Pool the result of the convolution with 2x2 non-overlapping neighborhoods
    //printf("Pooling\n");
    for(j = 0; j < curDepth; j++) {
      //This will also store the max indices in the proper blocks
      //printf("Feature map %d of %d: %dx%d\n", j, curDepth, curFeatureMaps[j].rows, curFeatureMaps[j].cols);
      if(i < net->numHiddenLayers-1) { //Need to do a bigger pooling operation before FC
	matrix_pool_storeIndices(&curFeatureMaps[j], 2, &max_idxs[i][j]);
      }
      //printf("Closed\n");
      /* printf("FEATURE MAP %d, %d\n", i, j); */
      /* matrix_print(&curFeatureMaps[j], stdout); */
      /* printf("end feature map %d, %d\n", i, j); */
      
      //If the next layer is convolutional, store the inputs to that layer
      //if(i != net->numHiddenLayers-1) {
      matrix_copy(&conv_x[i+1][j], &curFeatureMaps[j]);
	//}
    }


    if(i != 0) { //Free matrices from the previous layer
      for(j = 0; j < net->depths[i-1]; j++) {
	matrix_free(&prevFeatureMaps[j]);
      }
    }

    //Store the output of the current layer for operations from the next layer
    prevFeatureMaps = curFeatureMaps;
    prevDepth = curDepth;
    curFeatureMaps = NULL;

  }
	
  //Standard forward propagation from multi-layer perceptron network for fully connected layers

  /* First, make sure we only have one value from each feature map being passed into the
     fully connected layers */
  matrix_arrayToMaxMat_storeIndices(&metro1, prevFeatureMaps, curDepth, max_idxs[net->numHiddenLayers-1]);

  //Store the inputs to the fully connected layers for back propagation
  //append_ones(&metro1); //DEBUG
  matrix_copy(fullLayerInputs, &metro1);

  //printf("Entering fully complete layer\n");

  for(i = 0; i < net->numFullLayers + 1; i++) {
    if(i != net->numFullLayers) { //Account for biases in all but the output layer
      append_ones(&metro1);
    }
    metro2 = matrix_multiply_slow(&metro1, &(net->fullLayerWeights[i]));
    matrix_copy(&a[i], &metro2); //Store pre-activation
    if(i < net->numFullLayers) {
      tanh_matrix(&metro2);
      //append_ones(&metro2); //DEBUG
      matrix_copy(&h[i], &metro2); //Store output
    }
    else {
      //Use softmax non-linearity on the output layer
      softmax_matrix(&metro2);
    }

    matrix_copy(&metro1, &metro2);
    matrix_free(&metro2);

  }

  //Clean up
  for(i = 0; i < net->depths[net->numHiddenLayers-1]; i++) { //Matrices from the last pooling layer
    matrix_free(&prevFeatureMaps[i]);
  }
  
  return metro1;	
}

/* Subtracts all gradients from the current weights of the convolutional neural network */
void cnn_updateWeights(Conv_Neural_Network *net, Matrix **k_gradients, Matrix *fc_gradients, float learningRate) {
  int i, j, n, d;

  //Update convolution kernels
  for(n = 0; n < net->numHiddenLayers; n++) {
    for(d = 0; d < net->depths[n]; d++) {
      //printf("Layer %d; kernels %dx%d; gradients %dx%d\n", n, net->kernels[n][0].rows, net->kernels[n][0].cols, k_gradients[n][0].rows, k_gradients[n][0].cols);
      for(i = 0; i < k_gradients[n][d].rows; i++) {
	for(j = 0; j < k_gradients[n][d].cols; j++) {
	  net->kernels[n][d].m[i][j] -= (k_gradients[n][d].m[i][j] * learningRate);
	}
      }
    }
  }

  //Update fully connected layer weight matrices
  for(n = 0; n < net->numFullLayers+1; n++) {
    matrix_transpose(&fc_gradients[n]);
    //printf("Fully connected layer %d: weights %dx%d; gradients %dx%d\n", n, net->fullLayerWeights[n].rows, net->fullLayerWeights[n].cols, fc_gradients[n].rows, fc_gradients[n].cols);
    for(i = 0; i < fc_gradients[n].rows; i++) {
      for(j = 0; j < fc_gradients[n].cols; j++) {
	float t_max = matrix_max(&fc_gradients[n]);
	if(fabs(t_max) > 100) {
	  printf("Current gradient val at %d, %d, %d: %f\n", n, i, j, t_max);
	  exit(-1);
	}
	net->fullLayerWeights[n].m[i][j] -= (fc_gradients[n].m[i][j] * learningRate);
      }
    }
  }
}
