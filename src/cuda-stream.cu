/*
 * \file cuda-stream.c
 */

#include "cuda-stream.h"

cudaStream_t* createCudaStreams(unsigned int num)
{
  cudaStream_t* streams = (cudaStream_t*) calloc(num, sizeof(cudaStream_t));
  for (unsigned int i = 0; i < num; i++)
  {
    // TODO Do not ignore potential error.
    cudaError_t error = cudaStreamCreate(&streams[i]);
  }
  return streams;
}

void cleanCudaStreams(cudaStream_t* streams, unsigned int num)
{
  for (unsigned int i = 0; i < num; i++)
  {
    cudaStreamDestroy(streams[i]);
  }
}
