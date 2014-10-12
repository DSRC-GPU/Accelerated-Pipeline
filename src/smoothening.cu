/*
 * \file vector-smoothening.c
 */

#include "smoothening.h"
#include "util.h"
#include <stdio.h>


/*!
 * This smoothening function is not completely synchronized because it does not
 * use a global barrier.
 */
__global__ void smootheningRunKernel(float* xvectors,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    float phi, float* valuesOut)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  float values;
  if (gid < numvertices)
  {
    // DEBUG_PRINT("%u, %f\n", gid, xvectors[gid]);
    values = phi * xvectors[gid];
    for (size_t i = 0; i < numedges[gid]; i++)
    {
      unsigned int index = edges[gid + (numvertices * i)];
      values += ((1 - phi) * valuesOut[index]) / numedges[gid];
    }
  }
  __syncthreads();
  // if (gid == 0)
  //   DEBUG_PRINT("change: %f\n", xvectors[gid] - values);
  if (gid < numvertices)
    valuesOut[gid] = values;
}

void smootheningPrepareEdges(unsigned int* hostEdges,
    unsigned int* hostNumEdges, unsigned int totaledges,
    unsigned int totalvertices, unsigned int** edges, unsigned int** numedges)
{
  cudaMalloc(edges, totaledges * sizeof(unsigned int));
  cudaMalloc(numedges, totalvertices * sizeof(unsigned int));
  cudaMemcpy(*edges, hostEdges, totaledges * sizeof(unsigned int),
      cudaMemcpyHostToDevice);
  cudaMemcpy(*numedges, hostNumEdges, totalvertices * sizeof(unsigned int),
      cudaMemcpyHostToDevice);
}

void smootheningCleanEdges(unsigned int* edges, unsigned int* numedges)
{
  cudaFree(edges);
  cudaFree(numedges);
}

void smootheningRun(float* values,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    unsigned int numiterations, float phi, float* valuesOut)
{
  // Initialize the smoothened values as the input.
  cudaMemcpy(valuesOut, values, numvertices * sizeof(float),
      cudaMemcpyDeviceToDevice);

  unsigned int numblocks = ceil(numvertices / (float) BLOCK_SIZE);
  for (size_t i = 0; i < numiterations; i++)
  {
    cudaGetLastError();
    smootheningRunKernel<<<numblocks, BLOCK_SIZE>>>(values,
        numvertices, numedges, edges, phi, valuesOut);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("Cuda error: %s\n", cudaGetErrorString(err));
      exit (EXIT_FAILURE);
    }
  }

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  utilCudaCheckError(&err, "Error smoothening edges");
}

