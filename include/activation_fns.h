#ifndef ACTIVATION_FNS_H

#define ACTIVATION_FNS_H

#include "matrix.h"


/* Apply the sigmoid function to all elements in the input matrix mat */
void sigmoid_matrix(Matrix *mat);

/* Apply the sigmoid prime function to all elements in the input matrix mat */
void sigmoidPrime_matrix(Matrix *mat);

#endif
