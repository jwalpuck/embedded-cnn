#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "vector.h"

void test_matrices();
void test_vectors();

int main(int argc, char *argv[]) {
  //printf("\n\n\n---TESTING MATRICES---\n");
  test_matrices();
  //printf("\n\n\n---TESTING VECTORS---\n");
  test_vectors();
}

void test_matrices() {
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
  
  matrix_free(&m1);
  matrix_free(&m2);
  matrix_free(&m3);
}

void test_vectors() {
  Vector v1, v2;
  float *data1, *data2, dotprod;
  int length = 3;
  vector_init(&v1, length);
  vector_init(&v2, length);

  data1 = malloc(sizeof(float) * length);
  data2 = malloc(sizeof(float) * length);

  data1[0] = 8.2;
  data1[1] = 1.4;
  data1[2] = 0.7;

  data2[0] = 5.4;
  data2[1] = 6.1;
  data2[2] = 2.7;

  vector_fill(&v1, data1, length);
  vector_fill(&v2, data2, length);

  dotprod = dot(&v1, &v2);
  printf("The dot product of \n");
  vector_print(&v1, stdout);
  printf("and\n");
  vector_print(&v2, stdout);
  printf("is %f\n", dotprod);

  vector_free(&v1);
  vector_free(&v2);
  free(data1);
  free(data2);
}
