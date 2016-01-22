/**
 * vector.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include "vector.h"

/* Initialize an empty vector with zeros */
void vector_init(Vector *vec, int length) {
  vec->length = length;
  vec->v = calloc(length, sizeof(float));
}

/* Fill an empty vector with the given array */ 
void vector_fill(Vector *vec, float *data, int length) {
  vector_free(vec);
  vec->length = length;
  vec->v = data;
}

void vector_print(Vector *vec, FILE *fp) {
  if(fp) {
    int i;
    fprintf(fp, "<");
    for(i = 0; i < vec->length; i++) {
      fprintf(fp, "%f, ", vec->v[i]);
    }
    fprintf(fp, ">\n");
  }
}


/* Set the vector to all zeros */
void vector_clear(Vector *vec) {
  int i;
  for(i = 0; i < vec->length; i++) {
    vec->v[i] = 0;
  }
}

/* Free the dynamically allocated memory in a vector */
void vector_free(Vector *vec) {
  if(vec->v) {
    free(vec->v);
  }
}

/* Compute and return the dot product of vec1 and vec2 */
float dot(Vector *vec1, Vector *vec2) {
  int i;
  float prod = 0;
  for(i = 0; i < vec1->length; i++) {
    prod += vec1->v[i] * vec2->v[i];
  }
  return prod;
}

/* Multiply all elements in the vector by the parameterized scalar */
void scale(Vector *vec, float scalar) {
  int i;
  for(i = 0; i < vec->length; i++) {
    vec->v[i] *= scalar;
  }
}

/* Normalize the elements of the vector based on the max value in the vector */
void normalize(Vector *vec) {
  int i;
  
  //Find the max value of the vector
  float max = vec->v[0];
  for(i = 1; i < vec->length; i++) {
    if(vec->v[i] > max) {
      max = vec->v[i];
    }
  }

  //Divide all elements in the vector by the max
  for(i = 0; i < vec->length; i++) {
    vec->v[i] /= max;
  }
}
