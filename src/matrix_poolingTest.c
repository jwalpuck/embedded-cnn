/**
 * Jack Walpuck
 * matrix_poolingTest.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
  Matrix test;
  int i, j, rows, cols;
  float upperBound = 1.0;
  srand(time(NULL));

  rows = 2;
  cols = 2;

  matrix_init(&test, rows, cols);

  for(i = 0; i < test.rows; i++) {
    for(j = 0; j < test.cols; j++) {
      test.m[i][j] = (((float)rand()/(float)(RAND_MAX/upperBound)) * 2) - upperBound;
    }
  }

  printf("Before pooling:\n");
  matrix_print(&test, stdout);

  matrix_pool(&test, 2);

  printf("After pooling\n");
  matrix_print(&test, stdout);

  return 0;
}
