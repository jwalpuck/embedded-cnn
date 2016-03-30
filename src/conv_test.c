/*
 * conv_test.c
 * Jack Walpuck
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
  Matrix m, k, r0, r1, r2;
	
  matrix_init(&m, 3, 3);
  matrix_init(&k, 2, 2);
	
  //Populate matrices
  m.m[0][0] = 1;
  m.m[0][1] = 2;
  m.m[0][2] = 3;
  m.m[1][0] = 4;
  m.m[1][1] = 5;
  m.m[1][2] = 6;
  m.m[2][0] = 7;
  m.m[2][1] = 8;
  m.m[2][2] = 9;
	
  k.m[0][0] = 1;
  k.m[0][1] = 2;
  k.m[1][0] = 3;
  k.m[1][1] = 4;
	
  r0 = matrix_convolution(&m, &k);
  printf("Calculated standard convolution\n");
  r1 = matrix_tilde_convolution(&m, &k);
  printf("Calculated tilde convolution\n");
  r2 = matrix_zeroPad_convolution(&m, &k);
  printf("Calculated zero-padded convolution\n");
	
  printf("Result of convolution of:\n");
  matrix_print(&m, stdout);
  matrix_print(&k, stdout);
  printf("is\n");
  matrix_print(&r0, stdout);
  printf("Tilde convolution\n");
  matrix_print(&r1, stdout);
  printf("Zero-padded convolution\n");
  matrix_print(&r2, stdout);

  return 0;
}
