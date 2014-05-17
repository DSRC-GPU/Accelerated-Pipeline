
#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

typedef struct CudaTimer
{
  cudaEvent_t start, stop;
} CudaTimer;

void startCudaTimer(CudaTimer* cTimer);
void stopCudaTimer(CudaTimer* cTimer);
void resetCudaTimer(CudaTimer* cTimer);
void printCudaTimer(CudaTimer* cTimer);

#endif
