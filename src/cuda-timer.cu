#include "cuda-timer.h"
#include <stdio.h>

void startCudaTimer(CudaTimer* cTimer)
{
  cudaEventCreate(&cTimer->start);
  cudaEventRecord(cTimer->start, 0);
}

void stopCudaTimer(CudaTimer* cTimer)
{
  cudaEventCreate(&cTimer->stop);
  cudaEventRecord(cTimer->stop, 0);
  cudaEventSynchronize(cTimer->stop);
}

void resetCudaTimer(CudaTimer* cTimer)
{
  cudaEventDestroy(cTimer->start);
  cudaEventDestroy(cTimer->stop);
}

void printCudaTimer(CudaTimer* cTimer, char* msg)
{
  float time;
  cudaEventElapsedTime(&time, cTimer->start, cTimer->stop);
  printf("timer: %s\nkernel time (ms): %f.\n", msg, time);
}

