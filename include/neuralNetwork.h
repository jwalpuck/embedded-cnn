#ifndef NEURALNETWORK_H

#define NEURALNETWORK_H

#include "matrix.h"

typedef struct {
  int inputLayerSize;
  int outputLayerSize;
  int numHiddenLayers;
  int *hiddenLayerSizes;
  Matrix *weights;
} Neural_Network;

/* Initialize a neural network with the given structure */
void nn_init(Neural_Network *net, int numInputs, int numOutputs, int numHiddenLayers,
	  int *hiddenLayerSizes);

/* Free the dynamically allocated memory in the neural network */
void nn_free(Neural_Network *net);

/* Initialize a neural network with a pre-existing set of weights */
void nn_initWithWeights(Neural_Network *net, int numInputs, int numOutputs,
			int numHiddenLayers, int *hiddenLayerSizes, Matrix *weights);

/* Initialize the weight vectors with random floats */
void nn_generateWeights(Neural_Network *net);

/* Forward propogate inputs through the network and return the output vector */
Matrix nn_forward(Neural_Network *net, Matrix *inputs);

/* Forward propogate inputs through the network, return ouput and activity matrices
   from each layer */
Matrix nn_forward_activity(Neural_Network *net, Matrix *inputs, Matrix **z, Matrix **a);

/* Subtracts all gradients from the current weights of the neural network */
void nn_updateWeights(Neural_Network *net, Matrix *gradients, float learningRate);

#endif
