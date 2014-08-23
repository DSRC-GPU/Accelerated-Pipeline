
#include "connected-component.h"
#include "util.h"

__global__ void connectedComponentInitKernel(unsigned int numvertices,
    unsigned int* d_C, unsigned int* d_F1, unsigned int* d_F2)
{
  // Initialize arrays
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < numvertices)
  {
    d_C[gid] = gid;
    d_F1[gid] = 1;
    d_F2[gid] = 0;
  }
}

__global__ void connectedComponentKernel(unsigned int numvertices,
    unsigned int* numedges, unsigned int* edgeTargets,
    unsigned int* vertexlabels, unsigned int* f1, unsigned int* f2,
    unsigned int* m)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (f1[gid])
  {
    f1[gid] = 0;
    unsigned int c = vertexlabels[gid];
    unsigned int cmod = 0;
    for (size_t i = 0; i < numedges[gid]; i++)
    {
      unsigned int neighbourIndex = gid + i * numvertices;
      unsigned int neighbour = edgeTargets[neighbourIndex];
      unsigned int cneighbour = vertexlabels[neighbour];
      if (c < cneighbour)
      {
        atomicMin(&vertexlabels[neighbour], c);
        f2[neighbour] = 1;
        *m = 1;
      }
      else if (c > cneighbour)
      {
        c = cneighbour;
        cmod = 1;
      }
    }
    if (cmod)
    {
      atomicMin(&vertexlabels[gid], c);
      f2[gid] = 1;
      *m = 1;
    }
  }
}

void connectedComponent(unsigned int numvertices, unsigned int* numedges,
      unsigned int* edgeTargets, unsigned int* vertexlabels)
{
  unsigned int* d_F1 = NULL;
  unsigned int* d_F2 = NULL;

  cudaMalloc(&d_F1, numvertices * sizeof(unsigned int));
  cudaMalloc(&d_F2, numvertices * sizeof(unsigned int));

  unsigned int numblocks = ceil(numvertices / (float) BLOCK_SIZE);

  connectedComponentInitKernel<<<numblocks, BLOCK_SIZE>>>(numvertices,
      vertexlabels, d_F1, d_F2);

  unsigned int h_m = 1;
  unsigned int* d_m = NULL;
  cudaMalloc(&d_m, sizeof(unsigned int));

  do 
  {
    cudaMemset(d_m, 0, sizeof(unsigned int));
    connectedComponentKernel<<<numblocks, BLOCK_SIZE>>>(numvertices, numedges,
        edgeTargets, vertexlabels, d_F1, d_F2, d_m);

    unsigned int* tmp = d_F1;
    d_F1 = d_F2;
    d_F2 = tmp;

    cudaMemcpy(&h_m, d_m, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  } while (h_m);
}

