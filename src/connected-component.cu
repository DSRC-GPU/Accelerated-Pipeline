
#include "connected-component.h"
#include "util.h"

__global__ void connectedComponentKernel(unsigned int numvertices,
    unsigned int* numedges, unsigned int* edgeTargets,
    unsigned int* vertexlabels)
{
  // FIXME Implement.
}

void connectedComponent(unsigned int numvertices, unsigned int* numedges,
      unsigned int* edgeTargets, unsigned int* vertexlabels)
{
  unsigned int numblocks = ceil(numvertices / (float) BLOCK_SIZE);
  connectedComponentKernel<<<numblocks, BLOCK_SIZE>>>(numvertices, numedges,
      edgeTargets, vertexlabels);
}

