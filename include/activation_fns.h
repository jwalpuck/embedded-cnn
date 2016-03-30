#ifndef ACTIVATION_FNS_H

#define ACTIVATION_FNS_H

#include "matrix.h"


/* Apply the sigmoid function to all elements in the input matrix mat */
void sigmoid_matrix(Matrix *mat);

/* Apply the sigmoid prime function to all elements in the input matrix mat */
void sigmoidPrime_matrix(Matrix *mat);

/* Apply the tanh function to all elements in the input matrix mat */
void tanh_matrix(Matrix *mat);

/* Apply the tanh prime function to all elements in the input matrix mat:
   Eq: tanh(x)' = sec^2(x) */
void tanhPrime_matrix(Matrix *mat);

/* Apply the softmax function to all elements in the input matrix mat */
void softmax_matrix(Matrix *mat);

#endif
