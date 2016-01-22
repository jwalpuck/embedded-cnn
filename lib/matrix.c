/** 
 * Jack Walpuck
 * matrix.h
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"

//The size of a cache line
#define CLS 64
#define SM (CLS/sizeof(float))


/* Allocate a 2D array of size rowsxcols and fill it with zeros */
void matrix_init(Matrix *mat, int rows, int cols) {
  int j;
  
  mat->rows = rows;
  mat->cols = cols;
  
  //Create an empty 2D array within the matrix
  mat->m = malloc(sizeof(float *) * rows);
  for(j = 0; j < rows; j++) {
    mat->m[j] = calloc(cols, sizeof(float));
  }
}

/* Assign the fields of a matrix to given parameters */
void matrix_fill(Matrix *mat, float **data, int rows, int cols) {
  matrix_free(mat);
  mat->rows = rows;
  mat->cols = cols;
  mat->m = data;
}

/* Print out the matrix in a rowsxcols arrangement with blank line below */
void matrix_print(Matrix *mat, FILE *fp) {
  if(fp) {
    int i, j;
    fprintf(fp, "[");
    for(i = 0; i < mat->rows; i++) {
      for(j = 0; j < mat->cols; j++) {
	fprintf(fp, "%.9f, ", mat->m[i][j]);
      }
      if(i == mat->rows-1) {
	fprintf(fp, "]]\n\n");
      }
      else{
	fprintf(fp, "]\n");
      }
    }
  }
}

/* Set the matrix to all zeros */
void matrix_clear(Matrix *mat) {
  int i, j;
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      mat->m[i][j] = 0;
    }
  }
}

/* Frees all of the dynamically allocated memory in mat->m */
void matrix_free(Matrix *mat) {
  int i;
  if(mat->m){
    for(i = 0; i < mat->rows; i++) {
      free(mat->m[i]);
    }
    free(mat->m);
    mat->m = NULL;
  }
}

/* Return the element of the matrix at row r, column c */
float matrix_get(Matrix *mat, int r, int c) {
  return mat->m[r][c]; 
}

/* Set the element of the matrix at row r, column c to v */
void matrix_set(Matrix *mat, int r, int c, float v) {
  mat->m[r][c] = v;
}

/* Copy the src matrix into the dest matrix */
void matrix_copy(Matrix *dest, Matrix *src) {
  int i, j;

  matrix_free(dest);
  dest->rows = src->rows;
  dest->cols = src->cols;
  matrix_init(dest, dest->rows, dest->cols);
  
  for(i = 0; i < src->rows; i++) {
    for(j = 0; j < src->cols; j++) {
      dest->m[i][j] = src->m[i][j];
    }
  }
}

/* Transpose the SQUARE matrix m in place */
void matrix_transpose(Matrix *mat) {
  
  Matrix copy = emptyMatrix;
  int i, j;
  matrix_copy(&copy, mat);
  
  //Change the dimensions of mat to be nxm instead of mxn
  matrix_free(mat);
  matrix_init(mat, copy.cols, copy.rows);
  
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      mat->m[i][j] = copy.m[j][i];
    }
  }
}

/* Negate all elements in the parameterized matrix */
void matrix_negate(Matrix *mat) {
  int i, j;
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      mat->m[i][j] *= -1;
    }
  }
}

/* Multiply left and right and put the result in mat */
Matrix matrix_multiply_fast(Matrix *left, Matrix *right) {
  int i, j, k, i2, j2, k2;
  int N = right->cols;
  size_t fsize = sizeof(float);
  float *rmat, *rleft, *rright;
  Matrix mat;
  matrix_init(&mat, left->rows, right->cols);
  
  for(i = 0; i < left->rows; i += SM) {
    for(j = 0; j < N; j += SM) {
      for(k = 0; k < N; k += SM) {
	for(i2 = 0, rmat = &mat.m[i][j], rleft = &left->m[i][k];
	    i2 < SM;
	    ++i2, rmat += fsize, rleft += fsize) {
	  for(k2 = 0, rright = &right->m[k][j];
	      k2 < SM;
	      ++k2, rright += fsize) {
	    for(j2 = 0; j2 < SM; ++j2) {
	      rmat[j2] += rleft[k2] * rright[j2];
	    }
	  }
	}
      }
    }
  }
  return mat;
}


/* Multiply left and right and put the result in mat */
Matrix matrix_multiply_slow(Matrix *left, Matrix *right) {

  int i, j, k;
  Matrix temp = emptyMatrix, mat;
  matrix_init(&mat, left->rows, right->cols); //Matrix to return
  matrix_copy(&temp, right);
  matrix_transpose(&temp);
  
  for(i = 0; i < left->rows; i++) {
    for(j = 0; j < right->cols; j++) {
      for(k = 0; k < right->rows; k++) {
  	mat.m[i][j] += left->m[i][k] * temp.m[j][k];
      }
    }
  }

  return mat;
}

/* Perform element-wise multiplication between two matrices */
Matrix matrix_element_multiply(Matrix *left, Matrix *right) {
  int i, j;
  Matrix ret;
  matrix_init(&ret, left->rows, left->cols);
  
  for(i = 0; i < ret.rows; i++) {
    for(j = 0; j < ret.cols; j++) {
      ret.m[i][j] = left->m[i][j] * right->m[i][j];
    }
  }
  return ret;
}

/* Sum all of the elements of a matrix */
float matrix_sum(Matrix *mat) {
  int i, j;
  float sum = 0;
  for(i = 0; i < mat->rows; i++) {
    for(j = 0; j < mat->cols; j++) {
      sum += mat->m[i][j];
    }
  }
  return sum;
}

/* Element-wise subtraction of two mxn matrices */
Matrix matrix_subtract(Matrix *left, Matrix *right) {
  int i, j;
  Matrix ret;
  matrix_init(&ret, left->rows, left->cols);

  for(i = 0; i < ret.rows; i++) {
    for(j = 0; j < ret.cols; j++) {
      ret.m[i][j] = left->m[i][j] - right->m[i][j];
    }
  }
  return ret;
}