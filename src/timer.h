/*!
 * \file timer.h
 * Provides some simple methods to measure elapsed time.
 */
#ifndef TIMER_H
#define TIMER_H

/*!
 * Struct to save timing information.
 */
typedef struct Timer
{
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
} Timer;

/*!
 * Starts the timer.
 * \param t Timer to start.
 */
void startTimer(Timer* t);

/*!
 * Stops the timer.
 * \param t Timer to stop.
 */
void stopTimer(Timer* t);

/*!
 * Resets the timer.
 * \param t Timer to reset.
 */
void resetTimer(Timer* t);

/*!
 * Print the timer value. If the timer was stopped, print the time between start
 * and stop. If the timer was not stopped, prints the time between start and
 * now.
 * \param t Timer to print.
 */
void printTimer(Timer* t);

#endif
