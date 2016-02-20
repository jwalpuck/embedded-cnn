/** 
 * Jack Walpuck
 * matrix.h
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
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
      free(mat->m[i]); //For each row, free all of the columns
    }
    free(mat->m); //Free the array of rows
    mat->m = NULL;
  }
  mat->rows = 0;
  mat->cols = 0;
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
  //printf("Freed dest\n");
  //printf("About to initialize %dx%d matrix\n", src->rows, src->cols);
  matrix_init(dest, src->rows, src->cols);
  //printf("Initialized matrix\n");
  
  for(i = 0; i < src->rows; i++) {
    for(j = 0; j < src->cols; j++) {
      //printf("(%d,%d)\n", src->rows, src->cols);
      dest->m[i][j] = src->m[i][j];
    }
  }
}

/* Transpose the matrix mat in place */
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
  matrix_free(&copy);
}

/* Normalize each column in the matrix individually */
void matrix_normalize_columns(Matrix *mat) {
	int i, j;
	float max[mat->cols];
	
	//Initialize max values with first row
	for(j = 0; j < mat->cols; j++) {
		max[j] = mat->m[0][j];
	}
	
	//Search for max values
	for(i = 1; i < mat->rows; i++) {
		for(j = 0; j < mat->cols; j++) {
			if(mat->m[i][j] > max[j]) {
				max[j] = mat->m[i][j];
			}
		}
	}
	
	//Divide all entries in the matrix by their column's maximum
	for(i = 0; i < mat->rows; i++) {
		for(j = 0; j < mat->cols; j++) {
			mat->m[i][j] /= max[j];
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
  if(left->cols != right->rows) {
    printf("Dimensionality error: left %dx%d, right %dx%d\n", left->rows, left->cols, right->rows, right->cols);
    exit(-1);
  }

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
  matrix_free(&temp);
  
  return mat;
}

/* Perform element-wise multiplication between two matrices */
Matrix matrix_element_multiply(Matrix *left, Matrix *right) {
  if(!(left->rows == right->rows && left->cols == right->cols)) {
    printf("Error in element_multiply: left = %dx%d, right = %dx%d\n", left->rows, left->cols, right->rows, right->cols);
    exit(-1);
  }

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

/* Return an array of n matrices m = cur + (ratio*prev) where ratio is a scalar */
void matrix_momentum(Matrix *cur, Matrix *prev, int n, float ratio) {
	if(cur->rows != prev->rows || cur->cols != prev->cols) {
		printf("Error: gradients of differing size\n");
		exit(-1);
	}
	int i, j, count;
	
	for(count = 0; count < n; count++) { //For each matrix in the gradient array
		for(i = 0; i < cur->rows; i++) {
      for(j = 0; j < cur->cols; j++) {
				cur[count].m[i][j] += (prev[count].m[i][j] * ratio);
      }
    }
	}
}

/* Append a column of ones to the end of the given matrix (in place) */
void append_ones(Matrix *mat) {
  int i, j;
  Matrix temp = emptyMatrix;
  matrix_copy(&temp, mat);
  matrix_free(mat);

  matrix_init(mat, temp.rows, temp.cols+1);
  for(i = 0; i < temp.rows; i++) {
    for(j = 0; j < temp.cols+1; j++) {
      mat->m[i][j] = j != temp.cols ? temp.m[i][j] : 1;
    }
  }
  matrix_free(&temp);
}

/* Truncates the mat->rows'th row off of mat in place */
void matrix_truncate_row(Matrix *mat) {
	free(mat->m[mat->rows-1]);
	mat->rows--;
}

/* Returns the parameterized row of the given matrix as a 1xn matrix */
Matrix get_nth_row(Matrix *mat, int rowIdx) {
	int i;
	Matrix row;
	matrix_init(&row, 1, mat->cols);
	for(i = 0; i < mat->cols; i++) {
		row.m[0][i] = mat->m[rowIdx][i];
	}
	return row;
}

/* Given matrices m1 and m2 with the same number of rows, shuffle the rows of the two matrices in place,
 * maintaining the correspondence of row n in m1 to row n in m2 */
void matrix_shuffle_rows(Matrix *m1, Matrix *m2) {
	//Verify input
	if(m1->rows != m2->rows) {
		printf("Error: attempting to shuffle two matrices with different numbers of rows, exiting\n");
		exit(-1);
	}
	int i, j, randIdx, temp, numRows, *sequence;
	Matrix t1, t2; //Matrices to temporarily hold shuffle versions of m1 and m2
	
	numRows = m1->rows;
	sequence = malloc(sizeof(int) * numRows);
	matrix_init(&t1, m1->rows, m1->cols);
	matrix_init(&t2, m2->rows, m2->cols);
	
	//Randomize the order of the shuffle to be different each time
	srand(time(NULL));
	
	for(i = 0; i < numRows; i++) {
		sequence[i] = i;
	}
	
	//Shuffle the array of indices
	for(int i = numRows-1; i > 0; i--) {
		//Generate random index
		randIdx = rand()%i;
		
		//Swap array elements with each other
		temp = sequence[i];
		sequence[i] = sequence[randIdx];
		sequence[randIdx] = temp;
	}
	
	//Move the rows from the input matrices to the temp matrices given the randomized sequence of indices
	for(i = 0; i < numRows; i++) {
		for(j = 0; j < m1->cols; j++) {
			t1.m[i][j] = m1->m[sequence[i]][j];	
		}
		for(j = 0; j < m2->cols; j++) {
			t2.m[i][j] = m2->m[sequence[i]][j];
		}
	}
	
	//Copy the shuffled matrices back into their original structs
	matrix_copy(m1, &t1);
	matrix_copy(m2, &t2);
		
	//Clean up
	free(sequence);
	matrix_free(&t1);
	matrix_free(&t2);
}

/* Convolve matrix m2 over m1 and return the result in a new matrix */
Matrix matrix_convolution(Matrix *m1, Matrix *m2) {
	Matrix result;
	int i, j, sub1, sub2, r_rows, r_cols;
	
	r_rows = m1->rows - m2->rows + 1;
	r_cols = m1->cols - m2->cols + 1;
	matrix_init(&result, r_rows, r_cols);
	
	for(i = 0; i < r_rows; i++) {
		for(j = 0; j < r_cols; j++) {
			for(sub1 = 0; sub1 < m2->rows; sub1++) {
				for(sub2 = 0; sub2 < m2->cols; sub2++) {
					result.m[i][j] += m1->m[i+sub1][j+sub2] * m2->m[sub1][sub2];
				}
			}
		}
	}
	
	return result;
}

/* Pool the result of the convolution with dimxdim non-overlapping neighborhoods */
void matrix_pool(Matrix *mat, int dim) {
	Matrix temp;
	int rows, cols, i, j, sub1, sub2, max, cur;
	rows = mat->rows % 2 == 0 ? mat->rows / 2 : (mat->rows / 2) + 1;
	cols = mat->cols % 2 == 0 ? mat->cols / 2 : (mat->cols / 2) + 1;
	matrix_init(&temp, rows, cols);
	
	for(i = 0; i < temp.rows; i++) {
		for(j = 0; j < temp.cols; j++) {
			max = -9999;
			for(sub1 = 0; sub1 < dim; sub1++) {
				for(sub2 = 0; sub2 < dim; sub2++) {
					if(i+sub1 > temp.rows-1 || j+sub2 > temp.cols-1) {
						continue;
					}
					else { //Look for a new max
						cur = mat->m[i+sub1][j+sub2];
						if(cur > max) {
							max = cur;
						}
					}
				}
			}
			//After the 2x2 neighborhood has been traversed, assign the max value to the result matrix
			temp.m[i][j] = max;
		}
	}
	//Copy the new matrix into the given address
	matrix_copy(mat, &temp);
	
	//Clean up
	matrix_free(&temp);
}