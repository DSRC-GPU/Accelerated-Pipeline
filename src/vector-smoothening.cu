/*
 * \file vector-smoothening.c
 */

#include "vector-smoothening.h"
#include "util.h"

__global__ void vectorSmootheningRunKernel(float* xvectors, float* yvectors,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    float phi)
{
  unsigned int gid = threadIdx.x * (blockIdx.x * BLOCK_SIZE);
  float newvectorx, newvectory;
  if (gid < numvertices)
  {
    newvectorx = phi * xvectors[gid];
    newvectory = phi * yvectors[gid];
    for (size_t i = 0; i < numedges[gid]; i++)
    {
      unsigned int index = gid + numvertices * i;
      newvectorx += (1 - phi) * xvectors[index];
      newvectory += (1 - phi) * yvectors[index];
    }
  }
  __syncthreads();
  xvectors[gid] = newvectorx;
  yvectors[gid] = newvectory;
}

void vectorSmootheningRun(float* xvectors, float* yvectors,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    unsigned int numiterations, float phi)
{
  unsigned int numblocks = ceil(numvertices / (float) BLOCK_SIZE);
  for (size_t i = 0; i < numiterations; i++)
  {
    vectorSmootheningRunKernel<<<numblocks, BLOCK_SIZE>>>(xvectors, yvectors,
        numvertices, numedges, edges, phi);
  }
}
