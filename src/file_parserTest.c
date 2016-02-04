/*
 * file_parserTest.c
 * Jack Walpuck
 */
 
#include <stdio.h>
#include <string.h>
#include "matrix.h"
#include "file_parser.h"

int main(int argc, char *argv[]) {
	Matrix inputs, outputs;
	char *fileName;
	
	if(argc < 2) {
		printf("Usage: file_parserTest <fileName>\n:");
		return -1;
	}
	
	fileName = argv[1];
	file_to_matrix2(fileName, &inputs, &outputs);
	
	/* printf("Inputs:\n"); */
	/* matrix_print(&inputs, stdout); */
	/* printf("Outputs\n"); */
	/* matrix_print(&outputs, stdout); */

	matrix_free(&inputs);
	matrix_free(&outputs);
	
	return 0;
}
