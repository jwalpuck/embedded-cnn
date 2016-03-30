#ifndef CONVNEURALNET_H

#define CONVNEURALNET_H

#include "matrix.h"

typedef struct {
	int outputLayerSize;
	int numHiddenLayers;
	int numFullLayers;
	int *fullLayerSizes;
	int *kernelSizes;
	int *depths; //The number of neurons in each layer (number of feature maps)
	int stride; //How far should kernels move in between convolutions
	Matrix **kernels;
	Matrix *fullLayerWeights;
}Conv_Neural_Network;

/* Initialize a convolutional neural network with the given structure */
void cnn_init(Conv_Neural_Network *net, int numOutputs, int numHiddenLayers,
	  int numFullLayers, int *fullLayerSizes, int *kernelSizes, int *depths, int stride);

/* Free the dynamically allocated memory in the convolutional neural network */
void cnn_free(Conv_Neural_Network *net);

/* Initialize a convolutional neural network with a pre-existing set of weights */
void cnn_initWithWeights(Conv_Neural_Network *net, int numOutputs, int numHiddenLayers, 
													int numFullLayers, int *fullLayerSizes, int *kernelSizes, 
													int *depths, int stride, Matrix *weights);

/* Initialize the weight vectors with random floats */
void cnn_generateWeights(Conv_Neural_Network *net);

/* Forward propogate inputs through the network and return the output vector */
Matrix cnn_forward(Conv_Neural_Network *net, Matrix *inputs);

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
Matrix cnn_forward_activity(Conv_Neural_Network *net, Matrix *inputs, Matrix *fullLayerInputs, Matrix *a, Matrix *h, Matrix **y, Matrix **conv_x, Matrix **pool_x, Matrix **max_idxs);

/* Subtracts all gradients from the current weights of the convolutional neural network */
void cnn_updateWeights(Conv_Neural_Network *net, Matrix **k_gradients, Matrix *fc_gradients, float learningRate);
#endif
 
