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
__global__ void smootheningRunKernel(float* xvectors, float* yvectors,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    float phi, float* xvectorsOut, float* yvectorsOut)
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
      newvectorx += ((1 - phi) * xvectorsOut[index]) / numedges[gid];
      newvectory += ((1 - phi) * yvectorsOut[index]) / numedges[gid];
    }
  }
  __syncthreads();
  if (gid == 0)
    DEBUG_PRINT("change: %f\n", xvectors[gid] - newvectorx);
  xvectorsOut[gid] = newvectorx;
  yvectorsOut[gid] = newvectory;
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

void smootheningPrepareOutput(float** xoutput, float** youtput,
    unsigned int numvertices)
{
  cudaMalloc(xoutput, numvertices * sizeof(float));
  cudaMalloc(youtput, numvertices * sizeof(float));
}

void smootheningCleanEdges(unsigned int* edges, unsigned int* numedges)
{
  cudaFree(edges);
  cudaFree(numedges);
}

void smootheningRun(float* xvectors, float* yvectors,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    unsigned int numiterations, float phi, float* xvectorsOut, float* yvectorsOut)
{
  // Copy vectors. These will be the constant vectors used for smoothening. The
  // input array will be used for the smoothened values.
  cudaMemcpy(xvectorsOut, xvectors, numvertices * sizeof(float),
      cudaMemcpyDeviceToDevice);
  cudaMemcpy(yvectorsOut, yvectors, numvertices * sizeof(float),
      cudaMemcpyDeviceToDevice);

  unsigned int numblocks = ceil(numvertices / (float) BLOCK_SIZE);
  for (size_t i = 0; i < numiterations; i++)
  {
    cudaGetLastError();
    smootheningRunKernel<<<numblocks, BLOCK_SIZE>>>(xvectors,
        yvectors, numvertices, numedges, edges, phi, xvectorsOut, yvectorsOut);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("Cuda error: %s\n", cudaGetErrorString(err));
      exit (EXIT_FAILURE);
    }
  }
}
