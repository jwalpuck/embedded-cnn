#ifndef TRAINING_H

#define TRAINING_H

#include "matrix.h"
#include "neuralNetwork.h"
#include "conv_neuralNetwork.h"

/* Returns the cost of a neural network given inputs and corresponding outputs */
float cost_fn(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs);

/* Returns the overall cost via cross entropy of a convolutional neural network given
   a set of inputs and corresponding outputs of size n */
float cross_entropy(Conv_Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, int n);

/* Calculates the derivative of the cost function for the given neural network 
 * Note that a[0] = a_2, z[0] = z_2, weights[0] = W1, delta[0] = delta_2, n = total 
 * number of layers (input + hidden + output)
 */
Matrix *cost_fn_prime(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs);

/* Calculates the gradients for all kernels and weight matrices in net given a set of 
   inputs, their correct outputs, and a CNN to run them through */
void cnn_backProp(Conv_Neural_Network *net, Matrix *inputs, Matrix *correct_output, Matrix **kernel_gradients, Matrix *FC_gradients);

/* Calculates a new set of weights for the given Neural Network using the stochastic gradient
 * descent optimization algorithm. The network is trained on the parameterized matrices of inputs
 * and their corresponding correct outputs. The gradient will be recalculated n * numInputs times */						     
void stochastic_grad_descent(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, int n, float learningRate);

/* Calculates a new set of weights for the given CNN using back propagation with the given
   expected inputs and outputs. The gradient will be recalculated n * numInputs times.
   The parameterized inputs and outputs should be arrays of numInputs matrices. */
void cnn_stochastic_grad_descent(Conv_Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, int n, int numInputs, float learningRate);

/* Calculates a new set of weights for the given Neural Network using the stochastic gradient
 * descent optimization algorithm with momentum. The network is trained on the parameterized matrices 
 * of inputs and their corresponding correct outputs. The gradient will be recalculated n * numInputs times */
void stochastic_grad_descent_momentum(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs, 
																			int n, float learningRate, float momentumRatio);


#endif
