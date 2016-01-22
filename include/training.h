#ifndef TRAINING_H

#define TRAINING_H

#include "matrix.h"
#include "neuralNetwork.h"

/* Returns the cost of a neural network given inputs and corresponding outputs */
float cost_fn(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs);

Matrix *cost_fn_prime(Neural_Network *net, Matrix *inputs, Matrix *correct_outputs);

/* Returns an array of gradient matrices for the weights of the parameterized neural
 * network */
//Matrix *


#endif
