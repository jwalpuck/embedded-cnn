/*
 * matrix_test2.c
 * Jack Walpuck
 *
 * Testing memory operations of the matrix library
 */

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
  int i, j;
  Matrix test = emptyMatrix;
  Matrix test2 = emptyMatrix;
  Matrix *a = malloc(sizeof(Matrix) * 4);

  matrix_init(&test, 10, 10);
  matrix_copy(&test2, &test);

  printf("Freeing test\n");
  matrix_free(&test);
  printf("Freeing test2\n");
  matrix_free(&test2);

  for(i = 0; i < 4; i++) {
    matrix_init(&a[i], 40, 80);
  }
  
  for(i = 0; i < 4; i++) {
    printf("Freeing a[%d]\n", i);
    matrix_free(&a[i]);
  }

  printf("Freeing a\n");
  free(a);
}
