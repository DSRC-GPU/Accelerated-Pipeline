#include "timer.h"
#include <stdio.h>

/*!
 * Struct used to compute execution times in a parallel environmennt.
 */
typedef struct ParTimer {
  /*!
   * Struct to save the time when measuring is started.
   */
  cudaEvent_t start;
  /*!
   * Struct to save the time when measuring is ended.
   */
  cudaEvent_t stop;
} ParTimer;

Timer* timerNew()
{
  Timer* timer = calloc(1, sizeof(Timer));
  timer->internals = calloc(1, sizeof(ParTimer));
  return timer;
}

void timerClean(Timer* timer)
{
  if (timer)
  {
    free(timer->internals);
    free(timer);
  }
}

void startTimer(Timer* timer)
{
  ParTimer* ptimer = (ParTimer*) timer->internals;
  cudaEventCreate(&ptimer->start);
  cudaEventRecord(ptimer->start, 0);
}

void stopTimer(Timer* timer)
{
  ParTimer* ptimer = (ParTimer*) timer->internals;
  cudaEventCreate(&ptimer->stop);
  cudaEventRecord(ptimer->stop, 0);
  cudaEventSynchronize(ptimer->stop);
}

void resetTimer(Timer* timer)
{
  ParTimer* ptimer = (ParTimer*) timer->internals;
  cudaEventDestroy(ptimer->start);
  cudaEventDestroy(ptimer->stop);
}

void printTimer(Timer* timer, char* msg)
{
  ParTimer* ptimer = (ParTimer*) timer->internals;
  float time;
  cudaEventElapsedTime(&time, ptimer->start, ptimer->stop);
  printf("timer: %s\nkernel time (ms): %f\n", msg, time);
}

