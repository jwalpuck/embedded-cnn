#ifndef FILE_PARSER_H

#define FILE_PARSER_H

#include <string.h>
#include "matrix.h"

/* Parses the file with the given name and converts it into two matrices -- input and output 
 * File should be of the format: <input(s) separated by commas> : <output(s) separated by commas>
 * The matrices will be filled with rows equal to the number of lines in the file, and columns
 * equal to the number of comma-separated values in each row 
 */
void file_to_matrix(char *fileName, Matrix *inputs, Matrix *outputs);

/* Return the number of lines in the file at the parameterized path */
int count_lines(char *fileName);

/* Returns the address of an array of strings with each index containing a line from the input file */
void get_raw_text(char *fileName, char **raw_text, int numLines);

#endif