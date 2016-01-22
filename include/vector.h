#ifndef VECTOR_H

#define VECTOR_H

typedef struct {
  int length;
  float *v;
} Vector;

static const Vector emptyVector = {0, NULL};

/* Initialize an empty vector with zeros */
void vector_init(Vector *vec, int length);

/* Fill an empty vector with the given array */ 
void vector_fill(Vector *vec, float *data, int length);

/* Print out the vector with a blank line below */
void vector_print(Vector *vec, FILE *fp);

/* Set the vector to all zeros */
void vector_clear(Vector *vec);

/* Free the dynamically allocated memory in a vector */
void vector_free(Vector *vec);

/* Compute and return the dot product of vec1 and vec2 */
float dot(Vector *vec1, Vector *vec2);

/* Multiply all elements in the vector by the parameterized scalar */
void scale(Vector *vec, float scalar);

/* Normalize the elements of the vector based on the max value in the vector */
void normalize(Vector *vec);

#endif
