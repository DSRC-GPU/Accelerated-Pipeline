
#include "break-edges.h"
#include "util.h"

__device__ inline int sgnCmp(float a, float b)
{
  return a * b >= 0;
}

__global__ void breakEdgesKernel(unsigned int numVertices, float* fineValues,
    float* coarseValues, unsigned int* numEdges, unsigned int* edgeTargets)
{
  unsigned int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (gid < numVertices)
  {
    unsigned int numvertices = numEdges[gid];
    float localValue = fineValues[gid] - coarseValues[gid];
    for (size_t i = 0; i < numvertices; i++)
    {
      unsigned int index = numvertices * i + gid;
      unsigned int neighbour = edgeTargets[index];
      float neighbourValue = fineValues[neighbour] - coarseValues[neighbour];
      if (!sgnCmp(localValue, neighbourValue))
      {
        // Removing edge by setting target to itself.
        edgeTargets[neighbour] = gid;
      }
    }
  }
}

void breakEdges(unsigned int numVertices, float* fineValues,
    float* coarseValues, unsigned int* numEdges, unsigned int* edgeTargets)
{
  unsigned int numblocks = ceil(numVertices / (float) BLOCK_SIZE);
  breakEdgesKernel<<<numblocks, BLOCK_SIZE>>>(numVertices, fineValues,
      coarseValues, numEdges, edgeTargets);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  utilCudaCheckError(&err, "Error breaking edges");
}

