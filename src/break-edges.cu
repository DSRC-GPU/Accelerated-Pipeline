
#include "break-edges.h"
#include "util.h"

__global__ void breakEdgesKernel(unsigned int numVertices, float* fineValues,
    float* coarseValues, unsigned int* numEdges, unsigned int* edgeTargets)
{

}

void breakEdges(unsigned int numVertices, float* fineValues,
    float* coarseValues, unsigned int* numEdges, unsigned int* edgeTargets)
{
  unsigned int numblocks = ceil(numVertices / (float) BLOCK_SIZE);
  breakEdgesKernel<<<numblocks, BLOCK_SIZE>>>(numVertices, fineValues,
      coarseValues, numEdges, edgeTargets);
}

