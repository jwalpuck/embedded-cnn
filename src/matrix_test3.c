/*
 * matrix_test3.c
 * Jack Walpuck
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
	int i, j, rows;
	Matrix m1, m2;
	float **a1, **a2;
	
	m1 = emptyMatrix;
	m2 = emptyMatrix;
	
	printf("Allocating arrays\n");
	
	rows = 5;
	a1 = malloc(sizeof(float *) * rows);
	a2 = malloc(sizeof(float *) * rows);
	
	for(i = 0; i < rows; i++) {
		a1[i] = malloc(sizeof(float) * 2);
		a2[i] = malloc(sizeof(float) * 1);
	}
	
	for(i = 0; i < rows; i++) {
		for(j = 0; j < 2; j++) {
		//Fill a1
		a1[i][j] = (((float)rand()/(float)(RAND_MAX/5)) * 2) - 5;
			//Fill a2
			if(j < 1) {
				a2[i][j] = (((float)rand()/(float)(RAND_MAX/5)) * 2) - 5;
			}
		}
	}
	
	printf("Made arrays\n");
	
	matrix_fill(&m1, a1, rows, 2);
	printf("Filled m1\n");
	matrix_fill(&m2, a2, rows, 1);
	printf("Filled m2\n");
	
	printf("BEFORE shuffle:\n");
	printf("m1:\n");
	matrix_print(&m1, stdout);
	printf("m2:\n");
	matrix_print(&m2, stdout);
	
	//Shuffle the matrices
	matrix_shuffle_rows(&m1, &m2);
	
	printf("AFTER shuffle:\n");
	printf("m1:\n");
	matrix_print(&m1, stdout);
	printf("m2:\n");
	matrix_print(&m2, stdout);
	
	for(i = 0; i < rows; i++) {
		printf("Freeing row %d\n", i);
		free(a1[i]);
		free(a2[i]);
	}
	free(a1);
	free(a2);

	//Matrices do not need to be freed due to the (sloppy) nature of the implementation of matrix_fill
	
	return 0;
}