/*
 * matrix_test2.c
 * Jack Walpuck
 *
 * Testing memory operations of the matrix library
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
  int i, j;
  float randWeight, upperBound = 4.0;
  Matrix test = emptyMatrix;
  Matrix test2 = emptyMatrix;
  Matrix fac1 = emptyMatrix;
  Matrix fac2 = emptyMatrix;
  Matrix *a = malloc(sizeof(Matrix) * 4);

  matrix_init(&test, 10, 10);
  matrix_copy(&test2, &test);

  printf("Freeing test\n");
  matrix_free(&test);
  printf("Freeing test2\n");
  matrix_free(&test2);

  matrix_init(&fac1, 1000, 2);
  matrix_init(&fac2, 1000, 2);

  //Populate the matrices
  for(i = 0; i < 1000; i++) {
    for(j = 0; j < 2; j++) {
      //Generate a random weight in [0, upperBound]
      randWeight = ((float)rand()/(float)(RAND_MAX/upperBound) * 2) - upperBound;
      fac1.m[i][j] = randWeight;
      randWeight = ((float)rand()/(float)(RAND_MAX/upperBound) * 2) - upperBound;
      fac2.m[i][j] = randWeight;
    }
  }

  a[3] = matrix_element_multiply(&fac1, &fac2);

  /* for(i = 0; i < 4; i++) { */
  /*   matrix_init(&a[i], 40, 80); */
  /* } */
  
  /* for(i = 0; i < 4; i++) { */
  /*   printf("Freeing a[%d]\n", i); */
  /*   matrix_free(&a[i]); */
  /* } */

  printf("Freeing matrix\n");
  matrix_free(&a[3]);

  printf("Freeing a\n");
  free(a);
}
