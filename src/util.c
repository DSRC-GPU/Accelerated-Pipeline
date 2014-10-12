
#include <stdlib.h>
#include <stdio.h>

void utilVectorSetByScalar(float* dst, float scalar, unsigned int num)
{
  for (size_t i = 0; i < num; i++)
  {
    dst[i] = scalar;
  }
}

void utilVectorAdd(float* dst, float* src, unsigned int num)
{
  for (size_t i = 0; i < num; i++)
  {
    dst[i] += src[i];
  }
}

void utilVectorAddScalar(float* dst, float scalar, unsigned int num)
{
  for (size_t i = 0; i < num; i++)
  {
    dst[i] += scalar;
  }
}

void utilVectorAddInStream(float* dst, float* src, unsigned int num,
    void* stream)
{
  utilVectorAdd(dst, src, num);
}

void utilVectorMultiply(float* dst, float* src, unsigned int num)
{
  for (size_t i = 0; i < num; i++)
  {
    dst[i] *= src[i];
  }
}

void utilVectorMultiplyByScalar(float* dst, float scalar, unsigned int num)
{
  for (size_t i = 0; i < num; i++)
  {
    dst[i] *= scalar;
  }
}

void utilVectorDevide(float* dst, float* src, unsigned int num)
{
  for (size_t i = 0; i < num; i++)
  {
    dst[i] /= src[i];
  }
}

void utilVectorDevideByScalar(float* dst, float scalar, unsigned int num)
{
  for (size_t i = 0; i < num; i++)
  {
    dst[i] /= scalar;
  }
}

void utilParallelSum(float* d_M, unsigned int numelems, float* d_outVal)
{
  for (size_t i = 0; i < numelems; i++)
  {
    *d_outVal += d_M[i];
  }
}

void utilPrintDeviceArray(float* array, unsigned int numelems)
{
  for (size_t i = 0; i < numelems; i++)
  {
    printf("%lu, %f\n", i, array[i]);
  }
}

void* utilDataTransferHostToDevice(void* src, unsigned int numbytes,
    unsigned int freeHostMem)
{
  return src;
}

void* utilDataTransferDeviceToHost(void* src, unsigned int numbytes,
    unsigned int freeDeviceMem)
{
  return src;
}

void* utilAllocateData(unsigned int numbytes)
{
  return calloc(1, numbytes);
}

void utilFreeDeviceData(void* dptr)
{
  free(dptr);
}

void utilCudaCheckError(void* cudaError_t_ptr, char* msg)
{
  return;
}

