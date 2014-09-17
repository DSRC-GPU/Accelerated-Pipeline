
#include "vector-average.h"
#include "util.h"
#include <assert.h>
#include <stdlib.h>

float* vectorAverageNewVectorArray(unsigned int numelements)
{
  return (float*) calloc(2 * numelements, sizeof(float));
}

void vectorAverageFreeVectorArray(float* averageArray)
{
  free(averageArray);
}

float** vectorAverageNewWindow()
{
  return (float**) calloc(WINDOW_SIZE, sizeof(float*));
}

void vectorAverageFreeWindow(float** window)
{
  for (size_t i = 0; i < WINDOW_SIZE; i++)
  {
    free(window[i]);
  }
  free(window);
}

void vectorAverageShiftAndAdd(float** window, float* newEntry)
{
  free(window[0]);
  for (size_t i = 1; i < WINDOW_SIZE; i++)
  {
    window[i - 1] = window[i];
  }
  window[WINDOW_SIZE - 1] = newEntry;
}

void vectorAverageComputeAverage(float** window, unsigned int numelements,
    float* average)
{
  unsigned int total = WINDOW_SIZE;
  utilVectorSetByScalar(average, 0, numelements * 2); 
  for (size_t i = 0; i < WINDOW_SIZE; i++)
  {
    if (window[i])
      utilVectorAdd(average, window[i], numelements * 2);
    else
      total--;
  }
  assert(total != 0);
  utilVectorDevideByScalar(average, total, numelements * 2);
}

