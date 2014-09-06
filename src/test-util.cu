
#include "test-util.h"
#include "util.h"
#include <stdio.h>

void testUtil()
{
  unsigned int numelems = 1000;

  float* h_M = (float*) calloc(numelems, sizeof(float));
  for (size_t i = 0; i < numelems; i++)
  {
    h_M[i] = 1;
  }

  float* d_M = NULL;
  cudaMalloc(&d_M, numelems * sizeof(float));
  cudaMemcpy(d_M, h_M, numelems * sizeof(float), cudaMemcpyHostToDevice);

  float* d_outVal = NULL;
  cudaMalloc(&d_outVal, sizeof(float));

  utilParallelSum(d_M, numelems, d_outVal);

  float h_outVal = -1;
  cudaMemcpy(&h_outVal, d_outVal, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Sum: %f\n", h_outVal);
}

