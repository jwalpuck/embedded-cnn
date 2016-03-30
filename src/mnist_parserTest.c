/**
 * mnist_parserTest.c
 * Jack Walpuck
 */

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "mnist_parser.h"

int main(int argc, char *argv[]) {
  int n = 0;
  Matrix *images, *labels;
  char *img_filename, *lbl_filename;
  
  if(argc < 3) {
    printf("Usage: %s <mnist image filepath> <mnist label filepath>>\n", argv[0]);
    exit(-1);
  }

  img_filename = argv[1];
  lbl_filename = argv[2];

  images = read_mnist(img_filename, &n);
  printf("%d images read\n", n);
  printf("Sample matrix: %dx%d\n", images[n/2].rows, images[n/2].cols);
  matrix_print(&images[n/2], stdout);

  labels = read_mnist_labels(lbl_filename, &n);
  printf("%d labels read\n", n);
  printf("Sample label: %dx%d\n", labels[n/2].rows, labels[n/2].cols);
  matrix_print(&labels[n/2], stdout);
  

  return 0;
}
