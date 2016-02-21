/*
 * conv_test.c
 * Jack Walpuck
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
  int i, j, sub1, sub2, r_rows, r_cols;
  Matrix m, k, r;
	
  matrix_init(&m, 3, 3);
  matrix_init(&k, 2, 2);
  r_rows = m.rows-k.rows+1;
  r_cols = m.cols-k.cols+1;
  matrix_init(&r, m.rows-k.rows+1, m.cols-k.cols+1);
	
  //Populate matrices
  m.m[0][0] = 0;
  m.m[0][1] = 80;
  m.m[0][2] = 40;
  m.m[1][0] = 20;
  m.m[1][1] = 40;
  m.m[1][2] = 0;
  m.m[2][0] = 0;
  m.m[2][1] = 0;
  m.m[2][2] = 40;
	
  k.m[0][0] = 1;
  k.m[0][1] = 0.5;
  k.m[1][0] = .25;
  k.m[1][1] = 0;
	
  for(i = 0; i < r_rows; i++) {
    for(j = 0; j < r_cols; j++) {
      for(sub1 = 0; sub1 < k.rows; sub1++) {
	for(sub2 = 0; sub2 < k.cols; sub2++) {
	  r.m[i][j] += m.m[i+sub1][j+sub2] * k.m[sub1][sub2];
	}
      }
    }
  }
	
  printf("Result of convolution of:\n");
  matrix_print(&m, stdout);
  matrix_print(&k, stdout);
  printf("is\n");
  matrix_print(&r, stdout);

  return 0;
}
