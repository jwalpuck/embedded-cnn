#ifndef CONVNEURALNETWORK_H

#define CONVNEURALNETWORK_H

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
   from each layer */
Matrix cnn_forward_activity(Conv_Neural_Network *net, Matrix *inputs, Matrix *z, Matrix *a);

/* Subtracts all gradients from the current weights of the convolutional neural network */
void cnn_updateWeights(Conv_Neural_Network *net, Matrix *gradients, float learningRate);

#endif
 
