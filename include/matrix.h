#ifndef MATRIX_H

#define MATRIX_H
#define EMPTYMATRIX {0, 0, NULL}

typedef struct {
  int rows, cols;
  float **m;
} Matrix;

static const Matrix emptyMatrix = {0, 0, NULL};

/* Allocate a 2D array of size rowsxcols and fill it with zeros */
void matrix_init(Matrix *mat, int rows, int cols);

/* Assign the fields of a matrix to given parameters */
void matrix_fill(Matrix *mat, float **data, int rows, int cols);

/* Print out the matrix in a rowsxcols arrangement with blank line below */
void matrix_print(Matrix *mat, FILE *fp);

/* Set the matrix to all zeros */
void matrix_clear(Matrix *mat);

/* Frees all of the dynamically allocated memory in mat->m */
void matrix_free(Matrix *mat);

/* Return the element of the matrix at row r, column c */
float matrix_get(Matrix *mat, int r, int c);

/* Set the element of the matrix at row r, column c to v */
void matrix_set(Matrix *mat, int r, int c, float v);

/* Copy the src matrix into the dest matrix */
void matrix_copy(Matrix *dest, Matrix *src);

/* Transpose the matrix mat in place */
void matrix_transpose(Matrix *mat);

/* Normalize each column in the matrix individually */
void matrix_normalize_columns(Matrix *mat);

/* Negate all elements in the parameterized matrix */
void matrix_negate(Matrix *mat);

/* Multiply left and right and return the result */
Matrix matrix_multiply_fast(Matrix *left, Matrix *right);

/* Multiply left and right and return the result */
Matrix matrix_multiply_slow(Matrix *left, Matrix *right);

/* Perform element-wise multiplication between two matrices */
Matrix matrix_element_multiply(Matrix *left, Matrix *right);

/* Sum all of the elements of a matrix */
float matrix_sum(Matrix *mat);

/* Element-wise subtraction of two mxn matrices */
Matrix matrix_subtract(Matrix *left, Matrix *right);

/* Return an array of n matrices m = cur + (ratio*prev) where ratio is a scalar */
void matrix_momentum(Matrix *cur, Matrix *prev, int n, float ratio);

/* Append a column of ones to the end of the given matrix (in place) */
void append_ones(Matrix *mat);

/* Truncates the mat->rows'th row off of mat in place */
void matrix_truncate_row(Matrix *mat);

/* Returns the parameterized row of the given matrix as a 1xn matrix */
Matrix get_nth_row(Matrix *mat, int rowIdx);

/* Given matrices m1 and m2 with the same number of rows, shuffle the rows of the two matrices in place,
 * maintaining the correspondence of row n in m1 to row n in m2 */
void matrix_shuffle_rows(Matrix *m1, Matrix *m2);

/* Convolve matrix m2 over m1 and return the result in a new matrix */
Matrix matrix_convolution(Matrix *m1, Matrix *m2);

/* Pool the result of the convolution with dimxdim non-overlapping neighborhoods */
void matrix_pool(Matrix *mat, int dim);

#endif
