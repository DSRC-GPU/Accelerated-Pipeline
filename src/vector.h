
#ifndef VECTOR_H
#define VECTOR_H

#ifdef __CUDACC__
#define DEVICE_FUNC __device__
#else
#define DEVICE_FUNC
#endif

#include "graph.h"

DEVICE_FUNC void vectorAdd(double*, double*, double, double);
DEVICE_FUNC void vectorSubtract(double*, double*, double, double);
DEVICE_FUNC void vectorNormalize(double*, double*);
DEVICE_FUNC void vectorInverse(double*, double*);
DEVICE_FUNC void vectorMultiply(double*, double*, double);
DEVICE_FUNC void vectorCheckValid(double*, double*, char*);
DEVICE_FUNC double vectorGetLength(double, double);

#endif

