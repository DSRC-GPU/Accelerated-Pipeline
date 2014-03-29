
#ifndef VECTOR_H
#define VECTOR_H

#include "graph.h"

__device__ void vectorAdd(float*, float*, float, float);
__device__ void vectorSubtract(float*, float*, float, float);
__device__ void vectorNormalize(float*, float*);
__device__ void vectorInverse(float*, float*);
__device__ void vectorMultiply(float*, float*, float);
__device__ void vectorCheckValid(float*, float*, char*);
__device__ float vectorGetLength(float, float);

#endif

