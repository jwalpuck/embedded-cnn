/**
 * activation_fns.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
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
  float cur, e_term, denominator, debug;
  
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      cur = mat->m[i][j];
      
      debug = cur;
      
      if(isnan(cur)) {
	printf("Found nan  at (%d,%d) in matrix, exiting\n", i, j);
	exit(-1);
      }
      e_term = pow(M_E, -cur);
      denominator = (e_term+1) * (e_term+1);
      if(denominator == 0) {
	printf("Avoiding divide by 0\n");
	denominator = FLT_MIN;
      }
      mat->m[i][j] = e_term/denominator;
      //printf("new value: %f\n", mat->m[i][j]);
      if(isnan(mat->m[i][j])) {
	printf("****nan created at (%d,%d)\n", i, j);
	printf("original input was %f\n", debug);
      }
    }
  }
}

/* Apply the tanh function to all elements in the input matrix mat */
void tanh_matrix(Matrix *mat) {
  int i, j;

  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      mat->m[i][j] = tanh(mat->m[i][j]);
    }
  }
}

/* Apply the tanh prime function to all elements in the input matrix mat:
   Eq: tanh(x)' = sec^2(x) */
void tanhPrime_matrix(Matrix *mat) { 
  int i, j;
  float sec_term;
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      sec_term = (1/cosh(mat->m[i][j])) * (1/cosh(mat->m[i][j]));
      mat->m[i][j] = sec_term;
    }
  }
}

/* Apply the softmax function to all elements in the input matrix mat */
void softmax_matrix(Matrix *mat) {
  int i, j;
  float sum = 0;
  
  //First calculate the sum of all exp terms in the matrix
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      sum += pow(M_E, mat->m[i][j]);
    }
  }

  //Normalize
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      mat->m[i][j] = pow(M_E, mat->m[i][j]) / sum;
    }
  }
}
