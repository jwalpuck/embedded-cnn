/**
 * activation_fns.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "activation_fns.h"

/* Apply the sigmoid function to all elements in the input matrix mat */
void sigmoid_matrix(Matrix *mat) {
  int i, j;
  float cur, e_term;
  
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      cur = mat->m[i][j];
      e_term = pow(M_E, -cur);
      //Apply sigmoid function for the element cur
      mat->m[i][j] = (float)1/(float)(1+e_term);
    }
  }
}

/* Apply the sigmoid prime function to all elements in the input matrix mat */
void sigmoidPrime_matrix(Matrix *mat) {
  int i, j;
  float cur, e_term;
  
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      cur = mat->m[i][j];
      e_term = pow(M_E, -cur);
      mat->m[i][j] = e_term/(float)(1+(e_term * e_term));
    }
  }
}
