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
  int i, j, k, numLines, inputCols, outputCols;
  char ***split_colon, ***input_text, ***output_text, **raw_text, *running_start, *running, *token;
  inputCols = 0;
  outputCols = 0;
  running = NULL;
  running_start = NULL;
	
  //Read the raw text from the file
  numLines = count_lines(fileName);
  raw_text = malloc(sizeof(char *) * numLines);
  get_raw_text(fileName, raw_text, numLines);
	
  //Initialize the arrays for splitting strings with <numLines> rows
  split_colon = malloc(sizeof(char **) * numLines); //One array of strings for each line in the file
  input_text = malloc(sizeof(char **) * numLines);
  output_text = malloc(sizeof(char **) * numLines);
	
  //Split the lines at ':'
  for(i = 0; i < numLines; i++) {
    split_colon[i] = malloc(sizeof(char *) * 2); //One array for string on left side of colon, one for right
    running = strdup(raw_text[i]); //Create a duplicate of the input string to traverse
    running_start = running;
    for(j = 0; j < 3; j++) {
      token = strsep(&running, ":");
      if(token) {
				split_colon[i][j] = token;
      }
    }
    //free(running_start);
  }
	
  //How many columns in the input matrix?
  running = strdup(split_colon[0][0]);
  running_start = running;
  do {
    token = strsep(&running, ",");
    if(token) {
      inputCols++;
    }
  } while(token);
  //free(running_start);
	
  //How many columns in the output matrix?
  running = strdup(split_colon[0][1]);
  running_start = running;
  do {
    token = strsep(&running, ",");
    if(token) {
      outputCols++;
    }
  } while(token);
  //free(running_start);
	
  //Split the sub-lines at ','
  for(i = 0; i < numLines; i++) { //For each line in the file
    //For the left side of the colon (inputs)
    input_text[i] = malloc(sizeof(char *) * inputCols);
    running = strdup(split_colon[i][0]);
    running_start = running;
    if(inputCols > 1) {
      for(j = 0; j < inputCols; j++) {
	token = strsep(&running, ",");
	if(token) {
	  input_text[i][j] = token;
	}
      }
    }
    else {
      input_text[i][0] = running;
    }
    //free(running_start);
    
    //For the right side of the colon (outputs)
    output_text[i] = malloc(sizeof(char *) * outputCols);
    running = strdup(split_colon[i][1]);
    running_start = running;
    if(outputCols > 1) {
      for(k = 0; k < outputCols; k++) {
	token = strsep(&running, ",");
	if(token) {
	  output_text[i][k] = token;
	}
      }
    }
    else {
      output_text[i][0] = running;
    }
    //free(running_start);
  }
	
  //Fill the matrices, converting strings to numbers
  //printf("Initializing inputs\n");
  matrix_init(inputs, numLines, inputCols);
  //printf("Initializing outputs\n");
  matrix_init(outputs, numLines, outputCols);
	
  for(i = 0; i < numLines; i++) {
    for(j = 0; j < inputCols; j++) {
      if(input_text[i][j] == NULL) {
	printf("Read null input. Exiting\n");
	exit(-1);
      }
      inputs->m[i][j] = atof(input_text[i][j]);
    }
    for(k = 0; k < outputCols; k++) {
      if(output_text[i][k] == NULL) {
	printf("Read null output. Exiting\n");
	exit(-1);
      }
      outputs->m[i][k] = atof(output_text[i][k]);
    }
  }
	
  //Clean up
  for(i = 0; i < numLines; i++) {
    free(raw_text[i]);
    free(input_text[i]);
    free(output_text[i]);
    free(split_colon[i]);
  }
  free(raw_text);
  free(input_text);
  free(output_text);
  free(split_colon);
  free(running_start);
}

void file_to_matrix2(char *fileName, Matrix *inputs, Matrix *outputs) {
  printf("RECOMPILED2\n");
  int i, j, k, numLines, inputCols, outputCols;
  char ***split_colon, **raw_text, *running_start, *running, *token;
  inputCols = 0;
  outputCols = 0;
  running = NULL;
  running_start = NULL;
	
  //Read the raw text from the file
  numLines = count_lines(fileName);
  raw_text = malloc(sizeof(char *) * numLines);
  get_raw_text(fileName, raw_text, numLines);
	
  //Initialize the arrays for splitting strings with <numLines> rows
  split_colon = malloc(sizeof(char **) * numLines); //One array of strings for each line in the file
	
  //Split the lines at ':'
  for(i = 0; i < numLines; i++) {
    split_colon[i] = malloc(sizeof(char *) * 2); //One array for string on left side of colon, one for right
    running = strdup(raw_text[i]); //Create a duplicate of the input string to traverse
    running_start = running;
    for(j = 0; j < 3; j++) {
      token = strsep(&running, ":");
      if(token) {
	split_colon[i][j] = token;
      }
    }
    free(running_start);
  }
	
  //How many columns in the input matrix?
  running = strdup(split_colon[0][0]);
  running_start = running;
  do {
    token = strsep(&running, ",");
    if(token) {
      inputCols++;
    }
  } while(token);
  free(running_start);
	
  //How many columns in the output matrix?
  running = strdup(split_colon[0][1]);
  running_start = running;
  do {
    token = strsep(&running, ",");
    if(token) {
      outputCols++;
    }
  } while(token);
  free(running_start);

	
  //Fill the matrices, converting strings to numbers
  printf("Initializing inputs\n");
  matrix_init(inputs, numLines, inputCols);
  printf("Initializing outputs\n");
  matrix_init(outputs, numLines, outputCols);
  
  //Split the sub-lines at ','
  for(i = 0; i < numLines; i++) { //For each line in the file
    //For the left side of the colon (inputs)
    running = strdup(split_colon[i][0]);
    running_start = running;
    if(inputCols > 1) {
      for(j = 0; j < inputCols; j++) {
	token = strsep(&running, ",");
	if(token) {
	  inputs->m[i][j] = atof(token);
	}
      }
    }
    else {
      inputs->m[i][0] = atof(running);
    }
    free(running_start);
    
    //For the right side of the colon (outputs)
    running = strdup(split_colon[i][1]);
    running_start = running;
    if(outputCols > 1) {
      for(k = 0; k < outputCols; k++) {
	token = strsep(&running, ",");
	if(token) {
	  outputs->m[i][k] = atof(token);
	}
      }
    }
    else {
      outputs->m[i][0] = atof(running);
    }
    free(running_start);
  }
	
  //Clean up
  for(i = 0; i < numLines; i++) {
    free(raw_text[i]);
    free(split_colon[i]);
  }
  free(raw_text);
  free(split_colon);
}

/* Return the number of lines in the file at the parameterized path */
int count_lines(char *fileName) {
  FILE *input_file;
  int count = 0;
  char counter[1000];
	
  //Open file stream
  input_file = fopen(fileName, "r");
  if(input_file == 0) {
    perror("Cannot open input file, exiting\n");
    exit(-1);
  }
	
  //Read through the file once to figure out how many lines there are
  while(fgets(counter, 1000, input_file)) {
    count++;
  }
  fclose(input_file);
	
  return count;
}

/* Returns the address of an array of strings with each index containing a line from the input file */
void get_raw_text(char *fileName, char **raw_text, int numLines) {
  FILE *input_file;
  int i = 0;
	
  //Open file stream
  input_file = fopen(fileName, "r");
  if(input_file == 0) {
    perror("Cannot open input file, exiting\n");
    exit(-1);
  }
	
  //Copy text into the array
  for(i = 0; i < numLines; i++) {
    raw_text[i] = malloc(sizeof(char) * 1000);
    fgets(raw_text[i], 1000, input_file);
  }
	
  fclose(input_file);
}
