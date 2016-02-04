#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
  float **a1, **a2;
  int i, rows1, cols1, rows2, cols2;
  Matrix m1, m2, m3;
  m1 = emptyMatrix;
  m2 = emptyMatrix;
  m3 = emptyMatrix;

  //Create test matrices
  rows1 = 5;
  cols1 = 4;
  rows2 = 4;
  cols2 = 6;
  a1 = malloc(sizeof(float *) * rows1);
  a2 = malloc(sizeof(float *) * rows2);

  for(i = 0; i < rows1; i++) {
    a1[i] = calloc(cols1, sizeof(float));
  }
  for(i = 0; i < rows2; i++) {
    a2[i] = calloc(cols2, sizeof(float));
  }

  a1[2][3] = 18.09;
  a1[3][1] = 12.05;
  a1[1][3] = -8.99;
  a1[0][0] = 1.2;
  

  a2[0][1] = 1.0;
  a2[1][1] = 3.0;
  a2[2][1] = 7.0;
  a2[3][5] = 8.1;
  a2[0][4] = 16.2;
  
  matrix_fill(&m1, a1, rows1, cols1);
  matrix_fill(&m2, a2, rows2, cols2);

  printf("Matrices filled\n");
  
  matrix_print(&m1, stdout);
  matrix_print(&m2, stdout);

  m3 = matrix_multiply_slow(&m1, &m2);
  matrix_print(&m3, stdout);

  printf("Transposing m3\n");
  matrix_transpose(&m2);
  
  matrix_free(&m1); //Freeing too little?
  matrix_free(&m2);
  matrix_free(&m3);

  /* for(i = 0; i < rows1; i++) { */
  /*   free(a1[i]); //Freeing too little? */
  /* } */

  /* for(i = 0; i < rows2; i++) { */
  /*   free(a2[i]); */
  /* } */

  /* free(a1); */
  /* free(a2); */
}
