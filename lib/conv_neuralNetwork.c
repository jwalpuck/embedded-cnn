/*
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
  float randWeight, upperBound = 4.0;

  srand(time(NULL)); //COMMENT OUT FOR SAME SET OF RANDOM NUMBERS EACH TIME
  
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
	
	//Initialize weight matrix going into first complete layer (create an extra row for biases)
	matrix_init(&(net->fullLayerWeights[0]), net->depths[net->numHiddenLayers-1]+1, net->fullLayerSizes[0]);
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
	
}

/* Forward propogate inputs through the network and return the output vector */
Matrix cnn_forward(Conv_Neural_Network *net, Matrix *inputs){ 
	Matrix outputs, *curFeatureMaps;
	int i, j, d, curDepth;
	
	//Do depth[i-1] iterations over feature maps for convolutions, then sum all pre-activations before pooling
	//At i=0, only do one iteration (input is 1 dimensional)
	
	//Alternate convolution and pooling operations for numHiddenLayers iterations
	for(i = 0; i < net->numHiddenLayers; i++) {
		curDepth = net->depths[i];
		curFeatureMaps = malloc(sizeof(Matrix) * curDepth);
		
		//Perform a convolution for each feature map over input channels
		for(j = 0; j < curDepth; j++) {
			curFeatureMaps[j] = matrix_convolution(inputs, net->kernels[i][j]);
		}
		
		//Pool the result of the convolution with 2x2 non-overlapping neighborhoods
		for(j = 0; j < curDepth; j++) {
			curFeatureMaps[j] = matrix_pool(&(curFeatureMaps[j]), 2);
		}
	}
	
	//Standard forward propagation from multi-layer perceptron network for fully connected layers
	
	return outputs;
}

/* Forward propogate inputs through the network, return ouput and activity matrices
   from each layer */
Matrix cnn_forward_activity(Conv_Neural_Network *net, Matrix *inputs, Matrix *z, Matrix *a) {
	Matrix outputs;
	
	return outputs;
}

/* Subtracts all gradients from the current weights of the convolutional neural network */
void cnn_updateWeights(Conv_Neural_Network *net, Matrix *gradients, float learningRate) {

}