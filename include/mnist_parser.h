#ifndef MNISTPARSER_H

#define MNISTPARSER_H

#include "matrix.h"

/* Credit to https://compvisionlab.wordpress.com/ */
int reverseInt(int i);

/* Adapted from https://compvisionlab.wordpress.com */
Matrix *read_mnist(char *fileName, int *n);

Matrix *read_mnist_labels(char *fileName, int *n);

#endif
