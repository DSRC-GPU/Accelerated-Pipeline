
#include "timer.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*!
 * Struct used for computing execution times in sequential code.
 */
typedef struct SeqTimer {
  struct timespec start;
  struct timespec end;
} SeqTimer;

Timer* timerNew()
{
  Timer* timer = (Timer*) calloc(1, sizeof(Timer));
  timer->internals = calloc(1, sizeof(SeqTimer));
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

void startTimer(Timer* t)
{
  SeqTimer* timer = (SeqTimer*) t->internals;
  clock_gettime(CLOCK_MONOTONIC, &timer->start);
}

void stopTimer(Timer* t)
{
  SeqTimer* timer = (SeqTimer*) t->internals;
  clock_gettime(CLOCK_MONOTONIC, &timer->end);
}

void resetTimer(Timer* t)
{
  SeqTimer* timer = (SeqTimer*) t->internals;
  timer->start.tv_sec = 0;
  timer->start.tv_nsec = 0;
  timer->end.tv_sec = 0;
  timer->end.tv_nsec = 0;
}

void printTimer(Timer* t, char* msg)
{
  SeqTimer* timer = (SeqTimer*) t->internals;
  double ms = timer->end.tv_sec - timer->start.tv_sec;
  ms *= 1000;
  ms += (timer->end.tv_nsec - timer->start.tv_nsec)/1000000;
  printf("timer %s:\n%lf ms\n", msg, ms);
}

