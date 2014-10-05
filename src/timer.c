
#include "timer.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*!
 * Struct used for computing execution times in sequential code.
 */
typedef struct SeqTimer {
  /*!
   * Struct to save wall time start.
   */
  time_t wallTimerStart;
  /*!
   * Struct to save wall time end.
   */
  time_t wallTimerEnd;
  /*!
   * Struct to save cpu time start.
   */
  clock_t cpuTimerStart;
  /*!
   * Struct to save cpu time end.
   */
  clock_t cpuTimerEnd;
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
  timer->wallTimerEnd = 0;
  timer->cpuTimerEnd = 0;
  timer->wallTimerStart = time(NULL );
  timer->cpuTimerStart = clock();
}

void stopTimer(Timer* t)
{
  SeqTimer* timer = (SeqTimer*) t->internals;
  timer->wallTimerEnd = time(NULL );
  timer->cpuTimerEnd = clock();
}

void resetTimer(Timer* t)
{
  SeqTimer* timer = (SeqTimer*) t->internals;
  timer->wallTimerEnd = 0;
  timer->cpuTimerEnd = 0;
  timer->wallTimerStart = 0;
  timer->cpuTimerStart = 0;
}

void printTimer(Timer* t, char* msg)
{
  SeqTimer* timer = (SeqTimer*) t->internals;
  time_t wallEnd;
  clock_t cpuEnd;
  if (timer->wallTimerEnd != 0 || timer->cpuTimerEnd != 0)
  {
    wallEnd = timer->wallTimerEnd;
    cpuEnd = timer->cpuTimerEnd;
  }
  else
  {
    wallEnd = time(NULL );
    cpuEnd = clock();
  }
  //printf("Elapsed wall time (s):  %ld\n",
  //    (long) (wallEnd - timer->wallTimerStart));
  printf("timer: %s\ncpu time (ms):  %f\n", msg,
      (float) 1000 * (cpuEnd - timer->cpuTimerStart) / CLOCKS_PER_SEC);
}

