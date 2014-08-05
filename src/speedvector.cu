/*
 * \file speecvector.c
 */

#include "speedvector.h"
#include "util.h"

// Note: this could be done faster if each function would only call one kernel.

void speedVectorInit(float** averageSpeedX, float** averageSpeedY,
    float* vxLocs, float* vyLocs, unsigned int numvertices)
{
  cudaMalloc(averageSpeedX, numvertices);
  cudaMalloc(averageSpeedY, numvertices);
  cudaMemset(*averageSpeedX, 0, sizeof(float) * numvertices);
  cudaMemset(*averageSpeedY, 0, sizeof(float) * numvertices);
  utilVectorAdd(*averageSpeedX, vxLocs, numvertices);
  utilVectorMultiply(*averageSpeedX, vxLocs, numvertices);
  utilVectorAdd(*averageSpeedY, vyLocs, numvertices);
  utilVectorMultiply(*averageSpeedY, vyLocs, numvertices);
}

void speedVectorUpdate(float* vxLocs, float* vyLocs, float* averageSpeedX,
    float* averageSpeedY, unsigned int numvertices, cudaStream_t* stream)
{
  utilVectorAddInStream(averageSpeedX, vxLocs, numvertices, stream);
  utilVectorAddInStream(averageSpeedY, vyLocs, numvertices, stream);
}

void speedVectorFinish(float* averageSpeedX, float* averageSpeedY,
    unsigned int numiterations, unsigned int numvertices)
{
  utilVectorDevideByScalar(averageSpeedX, numiterations, numvertices);
  utilVectorDevideByScalar(averageSpeedY, numiterations, numvertices);
}

void speedVectorClean(float* averageSpeedX, float* averageSpeedY)
{
  cudaFree(averageSpeedX);
  cudaFree(averageSpeedY);
}
