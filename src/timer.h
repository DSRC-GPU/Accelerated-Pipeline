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
  void* internals;
} Timer;

/*!
 * Create a new timer.
 */
Timer* timerNew();

/*!
 * Clean up the timer.
 */
void timerClean(Timer* timer);

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
 * \param msg The message to print with the time.
 */
void printTimer(Timer* t, char* msg);

#endif
