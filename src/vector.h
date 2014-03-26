
#ifndef VECTOR_H
#define VECTOR_H

#include "graph.h"

void vectorAdd(float*, float*, float, float);
void vectorSubtract(float*, float*, float, float);
void vectorNormalize(float*, float*);
void vectorInverse(float*, float*);
void vectorMultiply(float*, float*, float);
void vectorCheckValid(float*, float*, char*);
float vectorGetLength(float, float);

#endif

