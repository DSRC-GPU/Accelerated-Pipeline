/*
 * \file vector-smoothening.c
 */

#include "vector-smoothening.h"
#include "util.h"
#include <stdio.h>

__global__ void vectorSmootheningRunKernel(float* xvectors, float* yvectors,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    float phi)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  float newvectorx, newvectory;
  if (gid < numvertices)
  {
    DEBUG_PRINT("%u, %f\n", gid, xvectors[gid]);
    newvectorx = phi * xvectors[gid];
    newvectory = phi * yvectors[gid];
    for (size_t i = 0; i < numedges[gid]; i++)
    {
      unsigned int index = edges[gid + (numvertices * i)];
      newvectorx += ((1 - phi) * xvectors[index]) / numedges[gid];
      newvectory += ((1 - phi) * yvectors[index]) / numedges[gid];
    }
  }
  __syncthreads();
  if (gid == 0)
    DEBUG_PRINT("change: %f\n", xvectors[gid] - newvectorx);
  xvectors[gid] = newvectorx;
  yvectors[gid] = newvectory;
}

void vectorSmootheningPrepareEdges(unsigned int* hostEdges,
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

void vectorSmootheningCleanEdges(unsigned int* edges, unsigned int* numedges)
{
  cudaFree(edges);
  cudaFree(numedges);
}

void vectorSmootheningRun(float* xvectors, float* yvectors,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    unsigned int numiterations, float phi)
{
  unsigned int numblocks = ceil(numvertices / (float) BLOCK_SIZE);
  for (size_t i = 0; i < numiterations; i++)
  {
    cudaGetLastError();
    vectorSmootheningRunKernel<<<numblocks, BLOCK_SIZE>>>(xvectors, yvectors,
        numvertices, numedges, edges, phi);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("Cuda error: %s\n", cudaGetErrorString(err));
      exit (EXIT_FAILURE);
    }
  }
}
