/* 
 * file_parser.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "file_parser.h"

/* Parses the file with the given name and converts it into two matrices -- input and output 
 * File should be of the format: <input(s) separated by commas> : <output(s) separated by commas>
 * The matrices will be filled with rows equal to the number of lines in the file, and columns
 * equal to the number of comma-separated values in each row 
 */
void file_to_matrix(char *fileName, Matrix *inputs, Matrix *outputs) {
	int i, j, k, l, numLines, inputCols, outputCols;
	char ***raw_text_ptr, ***split_colon, ***input_text, ***output_text, **raw_text, *running, *token, colon, comma;
	colon = ':';
	comma = ',';
	inputCols = 0;
	outputCols = 0;
	
	//Read the raw text from the file
	raw_text_ptr = malloc(sizeof(char **));
	raw_text = malloc(sizeof(char *));
	*raw_text_ptr = raw_text;
	printf("Address of raw_text = %p, %p\n", raw_text_ptr, &raw_text);
	printf("reading raw text\n");
	get_raw_text(fileName, raw_text_ptr, &numLines);
	printf("Popped get_raw_text off stack\n");
	printf("read raw text: line 1: %s\n", raw_text[0]);
	
	//Initialize the arrays for splitting strings with <numLines> rows
	split_colon = malloc(sizeof(char **) * numLines); //One array of strings for each line in the file
	input_text = malloc(sizeof(char **) * numLines);
	output_text = malloc(sizeof(char **) * numLines);
	
	printf("Allocated mid-level arrays\n");
	
	//Split the lines at ':'
	for(i = 0; i < numLines; i++) {
		printf("splitting line %d\n", i);
		split_colon[i] = malloc(sizeof(char *) * 2); //One array for string on left side of colon, one for right
		printf("Allocated sub-array\n");
		running = strdup(raw_text[i]); //Create a duplicate of the input string to traverse
		printf("Copied string %s\n", running);
		for(j = 0; j < 3; j++) {
			printf("On idx %d of separating at colon\n", j);
			token = strsep(&running, &colon);
			printf("token = %s\n", token);
			if(token) {
				split_colon[i][j] = token;
			}
		}
	}
	
	printf("Separated at colon\n");
	
	//How many columns in the input matrix?
	running = strdup(split_colon[0][0]);
	printf("Duplicated %s\n", running);
	do {
		token = strsep(&running, &comma);
		if(token) {
			inputCols++;
		}
	} while(token);
	
	printf("There are %d columns in the input matrix\n", inputCols);
	
	//How many columns in the output matrix?
	running = strdup(split_colon[0][1]);
	do {
		token = strsep(&running, &comma);
		if(token) {
			outputCols++;
		}
	} while(token);
	
	printf("There are %d columns in the output matrix\n", outputCols);
	
	//Split the sub-lines at ','
	for(i = 0; i < numLines; i++) { //For each line in the file
		//For the left side of the colon (inputs)
		input_text[i] = malloc(sizeof(char *) * inputCols);
		running = strdup(split_colon[i][0]);
		printf("Line %d input: %s\n", i, running);
		if(inputCols > 1) {
			for(j = 0; j < inputCols; j++) {
				token = strsep(&running, &comma);
				printf("input token %d = %s\n", j, token);
				if(token) {
					input_text[i][j] = token;
				}
			}
		}
		else {
			input_text[i][0] = running;
		}
		
		//For the right side of the colon (outputs)
		output_text[i] = malloc(sizeof(char *) * outputCols);
		running = strdup(split_colon[i][1]);
		printf("Line %d output: %s\n", i, running);
		if(outputCols > 1) {
			for(k = 0; k < outputCols; k++) {
				token = strsep(&running, &comma);
				printf("output token %d = %s\n", k, token);
				if(token) {
					output_text[i][k] = token;
				}
			}
		}
		else {
			output_text[i][0] = running;
		}
	}
	
	printf("Filling in matrices\n");
	
	//Fill the matrices, converting strings to numbers
	matrix_init(inputs, numLines, inputCols);
	matrix_init(outputs, numLines, outputCols);
	
	printf("matrices initialized\n");
	
	for(i = 0; i < numLines; i++) {
		printf("Filling row %d\n", i);
		for(j = 0; j < inputCols; j++) {
			printf("---\n");
			printf("Value %f\n", atof(input_text[i][j]));
			printf("---\n");
			inputs->m[i][j] = atof(input_text[i][j]);
		}
		for(k = 0; k < outputCols; k++) {
			printf("Made it inside output loop\n");
			printf("&&&\n");
			printf("Value %f\n", atof(output_text[i][k]));
			printf("&&&\n");
			outputs->m[i][k] = atof(output_text[i][k]);
		}
	}
	
	printf("Matrices filled\n");
	
	//Clean up
	for(i = 0; i < numLines; i++) {
		printf("Freeing line %d\n", i);
		free(raw_text[i]);
		free(input_text[i]);
		free(output_text[i]);
		free(split_colon[i]);
	}
	printf("Freeing raw_text_ptr\n");
	free(raw_text_ptr);
	printf("Raw_text_ptr freed\n");
}

/* Returns the address of an array of strings with each index containing a line from the input file */
void get_raw_text(char *fileName, char ***raw_text_ptr, int *numLines) {
	FILE *input_file;
	char **raw_text, counter[1000];
	int i = 0, localNumLines = 0;
	raw_text = *raw_text_ptr;
	//counter = malloc(sizeof(char *));
//	temp = NULL;
	
	printf("In get_raw_text: &raw_text = %p\n", raw_text);
	
	//Open file stream
	input_file = fopen(fileName, "r");
	if(input_file == 0) {
		perror("Cannot open input file, exiting\n");
		exit(-1);
	}
	
	printf("File stream opened\n");
	
	//Read through the file once to figure out how many lines there are
	while(fgets(counter, 1000, input_file)) {
		localNumLines++;
		printf("counted %d\n", localNumLines);
	}
	fclose(input_file);
	
	//Open file stream
	input_file = fopen(fileName, "r");
	if(input_file == 0) {
		perror("Cannot open input file, exiting\n");
		exit(-1);
	}
	
	//Copy text into the file
	//raw_text = malloc(sizeof(char *) * localNumLines);
	raw_text = realloc(raw_text, sizeof(char *) * localNumLines);
	for(i = 0; i < localNumLines; i++) {
		printf("Reading %d\n", i);
		raw_text[i] = malloc(sizeof(char) * 1000);
		fgets(raw_text[i], 1000, input_file);
		printf("read Line: %s\n", raw_text[i]);
	}
	
	fclose(input_file);
	
	//Store the number of lines in the file
	*numLines = localNumLines;
	printf("NumLines = %d\n", *numLines);
}