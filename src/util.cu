/*
 * \file util.cu
 */

#include "util.h"
#include "stdio.h"

__global__ void utilVectorSetByScalarKernel(float* dst, float scalar,
    unsigned int num)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < num)
    dst[gid] = scalar;
}

__global__ void utilVectorAddKernel(float* dst, float* src, unsigned int num)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < num)
    dst[gid] += src[gid];
}

__global__ void utilVectorMultiplyKernel(float* dst, float* src,
    unsigned int num)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < num)
    dst[gid] *= src[gid];
}

__global__ void utilVectorMultiplyByScalarKernel(float* dst, float scalar,
    unsigned int num)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < num)
    dst[gid] *= scalar;
}

__global__ void utilVectorDevideKernel(float* dst, float* src, unsigned int num)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < num)
    dst[gid] /= src[gid];
}

__global__ void utilVectorDevideByScalarKernel(float* dst, float denumerator,
    unsigned int num)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < num)
    dst[gid] /= denumerator;
}

void utilVectorSetByScalar(float* dst, float scalar, unsigned int num)
{
  unsigned int numblocks = ceil(num / (float) BLOCK_SIZE);
  utilVectorSetByScalarKernel<<<numblocks, BLOCK_SIZE>>>(dst, scalar, num);
}

void utilVectorAddInStream(float* dst, float* src, unsigned int num,
    cudaStream_t* stream)
{
  unsigned int numblocks = ceil(num / (float) BLOCK_SIZE);
  utilVectorAddKernel<<<numblocks, BLOCK_SIZE, 0, *stream>>>(dst, src, num);
}

void utilVectorAdd(float* dst, float* src, unsigned int num)
{
  unsigned int numblocks = ceil(num / (float) BLOCK_SIZE);
  utilVectorAddKernel<<<numblocks, BLOCK_SIZE>>>(dst, src, num);
}

void utilVectorMultiply(float* dst, float* src, unsigned int num)
{
  unsigned int numblocks = ceil(num / (float) BLOCK_SIZE);
  utilVectorMultiplyKernel<<<numblocks, BLOCK_SIZE>>>(dst, src, num);
}

void utilVectorMultiplyByScalar(float* dst, float scalar, unsigned int num)
{
  unsigned int numblocks = ceil(num / (float) BLOCK_SIZE);
  utilVectorMultiplyByScalarKernel<<<numblocks, BLOCK_SIZE>>>(dst, scalar, num);
}

void utilVectorDevide(float* dst, float* src, unsigned int num)
{
  unsigned int numblocks = ceil(num / (float) BLOCK_SIZE);
  utilVectorDevideKernel<<<numblocks, BLOCK_SIZE>>>(dst, src, num);
}

void utilVectorDevideByScalar(float* dst, float scalar, unsigned int num)
{
  unsigned int numblocks = ceil(num / (float) BLOCK_SIZE);
  utilVectorDevideByScalarKernel<<<numblocks, BLOCK_SIZE>>>(dst, scalar, num);
}
