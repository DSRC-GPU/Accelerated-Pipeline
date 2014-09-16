/*
 * \file util.cu
 */

#include "util.h"
#include "stdio.h"
#include <assert.h>

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

__global__ void utilVectorAddScalarKernel(float* dst, float scalar, unsigned int num)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < num)
    dst[gid] += scalar;
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

__global__ void utilParallelSumKernel(float* d_inM, unsigned int numelems,
    float* d_outVal)
{
  __shared__ float scratch[BLOCK_SIZE * 2];

  // Setup local data to perform reduction.
  unsigned int index = threadIdx.x;
  unsigned int base = index + (blockIdx.x * BLOCK_SIZE * 2);
  unsigned int stride = BLOCK_SIZE;

  if (base < numelems)
  {
    scratch[index] = d_inM[base];
  }
  else
    scratch[index] = 0;

  if (base + stride < numelems)
  {
    scratch[index + stride] = d_inM[base + stride];
  }
  else
    scratch[index + stride] = 0;

  // Do block-local reduction.
  while (stride > 0)
  {
    __syncthreads();
    if (index < stride)
    {
      scratch[index] += scratch[index + stride];
    }

    stride >>= 1;
  }

  // Do atomic add per block to obtain final value.
  __syncthreads();
  if (index == 0)
    atomicAdd(d_outVal, scratch[index]);
}

__global__ void utilPrintDeviceArrayKernel(float* array, unsigned int numelems)
{
  unsigned int gid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  if (gid < numelems)
    printf("%u, %f\n", gid, array[gid]);
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

void utilVectorAddScalar(float* dst, float scalar, unsigned int num)
{
  unsigned int numblocks = ceil(num / (float) BLOCK_SIZE);
  utilVectorAddScalarKernel<<<numblocks, BLOCK_SIZE>>>(dst, scalar, num);
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

void utilParallelSum(float* d_M, unsigned int numelems, float* d_outVal)
{
  unsigned int numblocks = ceil(numelems / ((float) BLOCK_SIZE * 2));
  cudaMemset(d_outVal, 0, sizeof(float));
  utilParallelSumKernel<<<numblocks, BLOCK_SIZE>>>(d_M, numelems, d_outVal);
}

void utilPrintDeviceArray(float* array, unsigned int numelems)
{
  unsigned int numblocks = ceil(numelems / (float) BLOCK_SIZE);
  printf("Device Array Print\n");
  cudaDeviceSynchronize();
  utilPrintDeviceArrayKernel<<<numblocks, BLOCK_SIZE>>>(array, numelems);
  cudaDeviceSynchronize();
}

void* utilDataTransferHostToDevice(void* src, unsigned int
    numbytes, unsigned int freeHostMem)
{
  void* dst = NULL;
  cudaMalloc(&dst, numbytes);
  cudaMemcpy(dst, src, numbytes, cudaMemcpyHostToDevice);
  if (freeHostMem)
    free(src);
  return dst;
}

void* utilDataTransferDeviceToHost(void* src, unsigned int numbytes,
    unsigned int freeDeviceMem)
{
  void* dst = calloc(1, numbytes);
  cudaMemcpy(dst, src, numbytes, cudaMemcpyDeviceToHost);
  if (freeDeviceMem)
    cudaFree(src);
  return dst;
}

void* utilAllocateData(unsigned int numbytes)
{
  void* res = NULL;
  cudaError_t err = cudaMalloc(&res, numbytes);
  assert(err != cudaErrorMemoryAllocation);
  return res;
}

void utilFreeDeviceData(void* dptr)
{
  cudaFree(dptr);
}

void utilCudaCheckError(void* cudaError_t_ptr, char* msg)
{
  cudaError_t* err = (cudaError_t*) cudaError_t_ptr;
  if (*err != cudaSuccess)
  {
    printf("Cuda error:\n%s\n%s\n", msg, cudaGetErrorString(*err));
  }
}
