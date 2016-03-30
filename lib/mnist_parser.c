/*
 * mnist_parser.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "mnist_parser.h"

/* Credit to https://compvisionlab.wordpress.com/ */
int reverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1=i&255;
  ch2=(i>>8)&255;
  ch3=(i>>16)&255;
  ch4=(i>>24)&255;
  return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

/* Adapted from https://compvisionlab.wordpress.com */
Matrix *read_mnist(char *fileName, int *n) {
  int i, r, c, rows, cols, magic_number;
  unsigned char temp;
  FILE *input_file;
  size_t size;
  Matrix *dest;

  magic_number = 0;
  *n = 0;
  rows = 0;
  cols = 0;

  //Open file stream
  input_file = fopen(fileName, "rb");
  if(input_file == 0) {
    perror("Cannot open input file, exiting\n");
    exit(-1);
  }

  //Read metadata
  size = fread((void *)&magic_number, sizeof(magic_number), 1, input_file);
  magic_number = reverseInt(magic_number);
  
  size = fread((void *)n, sizeof(*n), 1, input_file);
  *n = reverseInt(*n);
  
  size = fread((void *)&rows, sizeof(rows), 1, input_file);
  rows = reverseInt(rows);

  size = fread((void *)&cols, sizeof(cols), 1, input_file);
  cols = reverseInt(cols);

  //Allocate matrices
  dest = malloc(sizeof(Matrix) * (*n));
  for(i = 0; i < *n; i++) {
    matrix_init(&dest[i], rows, cols);
  }

  for(i = 0; i < *n; i++) {
    for(r = 0; r < rows; r++) {
      for(c = 0; c < cols; c++) {
	temp = 0;
	size = fread((char *)&temp, sizeof(temp), 1, input_file);
	dest[i].m[r][c]= (float)temp;
      }
    }
  }

  //Close the file stream
  fclose(input_file);

  //Make the compiler happy
  printf("%zu\n", size);

  return dest;
}

Matrix *read_mnist_labels(char *fileName, int *n) {
  int i, magic_number;
  unsigned char temp;
  FILE *input_file;
  size_t size;
  Matrix *dest;

  magic_number = 0;
  *n = 0;

  //Open file stream
  input_file = fopen(fileName, "rb");
  if(input_file == 0) {
    perror("Cannot open input file, exiting\n");
    exit(-1);
  }

  //Read metadata
  size = fread((void *)&magic_number, sizeof(magic_number), 1, input_file);
  magic_number = reverseInt(magic_number);
  
  size = fread((void *)n, sizeof(*n), 1, input_file);
  *n = reverseInt(*n);

  //Allocate matrices
  dest = malloc(sizeof(Matrix) * (*n));
  for(i = 0; i < *n; i++) {
    matrix_init(&dest[i], 1, 10); //Classes for digits 0-9;
  }

  for(i = 0; i < *n; i++) {
    temp = 0;
    size = fread((char *)&temp, sizeof(temp), 1, input_file);
    dest[i].m[0][(int)temp] = 1.0;
  }
  
  //Close the file stream
  fclose(input_file);

  //Make the compiler happy
  printf("%zu\n", size);

  return dest;
}
