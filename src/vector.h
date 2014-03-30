
#ifndef VECTOR_H
#define VECTOR_H

#ifdef __CUDACC__
#define DEVICE_FUNC __device__
#else
#define DEVICE_FUNC
#endif

#include "graph.h"

DEVICE_FUNC void vectorAdd(float*, float*, float, float);
DEVICE_FUNC void vectorSubtract(float*, float*, float, float);
DEVICE_FUNC void vectorNormalize(float*, float*);
DEVICE_FUNC void vectorInverse(float*, float*);
DEVICE_FUNC void vectorMultiply(float*, float*, float);
DEVICE_FUNC void vectorCheckValid(float*, float*, char*);
DEVICE_FUNC float vectorGetLength(float, float);

#endif

