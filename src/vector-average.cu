
#include "vector-average.h"
#include "util.h"
#include <assert.h>
#include <stdio.h>

float* vectorAverageNewVectorArray(unsigned int numelements)
{
  float* array = NULL;
  cudaError_t err = cudaMalloc(&array, 2 * numelements * sizeof(float));
  assert(err != cudaErrorMemoryAllocation);
  return array;
}

void vectorAverageFreeVectorArray(float* averageArray)
{
  cudaFree(averageArray);
}

float** vectorAverageNewWindow()
{
  return (float**) calloc(WINDOW_SIZE, sizeof(float*));
}

void vectorAverageFreeWindow(float** window)
{
  for (size_t i = 0; i < WINDOW_SIZE; i++)
  {
    cudaFree(window[i]);
  }
  free(window);
}

void vectorAverageShiftAndAdd(float** window, float* newEntry)
{
  cudaFree(window[0]);
  for (size_t i = 1; i < WINDOW_SIZE; i++)
  {
    window[i - 1] = window[i];
  }
  window[WINDOW_SIZE - 1] = newEntry;
  for (size_t i = 0; i < WINDOW_SIZE; i++)
  {
    DEBUG_PRINT("Window [%lu]: %p\n", i, window[i]);
  }
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

