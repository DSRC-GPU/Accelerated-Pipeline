#include <stdio.h>
#include <time.h>
#include "timer.h"

void startTimer(Timer* t)
{
  t->wallTimerEnd = 0;
  t->cpuTimerEnd = 0;
  t->wallTimerStart = time(NULL );
  t->cpuTimerStart = clock();
}

void stopTimer(Timer* t)
{
  t->wallTimerEnd = time(NULL );
  t->cpuTimerEnd = clock();
}

void resetTimer(Timer* t)
{
  t->wallTimerEnd = 0;
  t->cpuTimerEnd = 0;
  t->wallTimerStart = 0;
  t->cpuTimerStart = 0;
}

void printTimer(Timer* t)
{
  time_t wallEnd;
  clock_t cpuEnd;
  if (t->wallTimerEnd != 0 || t->cpuTimerEnd != 0)
  {
    wallEnd = t->wallTimerEnd;
    cpuEnd = t->cpuTimerEnd;
  }
  else
  {
    wallEnd = time(NULL );
    cpuEnd = clock();
  }
  printf("Elapsed wall time (s):  %ld\n", (long) (wallEnd - t->wallTimerStart));
  printf("Elapsed cpu time (ms):  %f\n",
      (float) 1000 * (cpuEnd - t->cpuTimerStart) / CLOCKS_PER_SEC);
}

