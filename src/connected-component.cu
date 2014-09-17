
#include "connected-component.h"
#include "util.h"
#include "stdio.h"

__global__ void connectedComponentInitKernel(unsigned int numvertices,
    unsigned int* d_C, unsigned int* d_F1, unsigned int* d_F2)
{
  // Initialize arrays
  unsigned int gid = BLOCK_SIZE * blockIdx.x + threadIdx.x;
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
  unsigned int gid = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  if (gid < numvertices)
  {
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
}

void connectedComponent(unsigned int numvertices, unsigned int* numedges,
      unsigned int* edgeTargets, unsigned int* vertexlabels)
{
  unsigned int* d_F1 = NULL;
  unsigned int* d_F2 = NULL;

  cudaError_t m1err = cudaMalloc(&d_F1, numvertices * sizeof(unsigned int));
  cudaError_t m2err = cudaMalloc(&d_F2, numvertices * sizeof(unsigned int));
  utilCudaCheckError(&m1err, "Error allocating F1");
  utilCudaCheckError(&m2err, "Error allocating F2");

  unsigned int numblocks = ceil(numvertices / (float) BLOCK_SIZE);
  
  cudaDeviceSynchronize();
  cudaGetLastError();
  connectedComponentInitKernel<<<numblocks, BLOCK_SIZE>>>(numvertices,
      vertexlabels, d_F1, d_F2);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  utilCudaCheckError(&err, "Error in init kernel.");

  unsigned int* h_m = (unsigned int*) calloc(1, sizeof(unsigned int));
  unsigned int* d_m = NULL;
  cudaMalloc(&d_m, sizeof(unsigned int));

  *h_m = 1;
  while (*h_m)
  {
    cudaMemset(d_m, 0, sizeof(unsigned int));

    connectedComponentKernel<<<numblocks, BLOCK_SIZE>>>(numvertices, numedges,
        edgeTargets, vertexlabels, d_F1, d_F2, d_m);

    unsigned int* tmp = d_F1;
    d_F1 = d_F2;
    d_F2 = tmp;

    cudaMemcpy(h_m, d_m, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  }

  cudaFree(d_F1);
  cudaFree(d_F2);
}

