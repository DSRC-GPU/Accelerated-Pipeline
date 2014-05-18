/*!
 * \file cuda-timer.h
 * Provides functions to time cuda operations.
 */

#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

/*!
 * structure to save timing information.
 */
typedef struct CudaTimer
{
  cudaEvent_t start, stop;
} CudaTimer;

/*!
 * Start the CudaTimer.
 */
void startCudaTimer(CudaTimer* cTimer);

/*!
 * Stops the CudaTimer.
 */
void stopCudaTimer(CudaTimer* cTimer);

/*!
 * Resets the CudaTimer to 0.
 */
void resetCudaTimer(CudaTimer* cTimer);

/*!
 * Prints the elapsed time between timer start and timer stop.
 */
void printCudaTimer(CudaTimer* cTimer);

#endif
