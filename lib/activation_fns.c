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
