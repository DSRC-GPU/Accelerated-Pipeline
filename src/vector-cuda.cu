
#include <stdio.h>
#include "vector-cuda.h"
#include "graph.h"

#define FLOAT_EPSILON 0.0000001

__device__ void vectorAdd(float* vxptr, float* vyptr, float vx, float vy)
{
  *vxptr += vx;
  *vyptr += vy;
}

__device__ void vectorSubtract(float* vxptr, float* vyptr, float vx, float vy)
{
  *vxptr -= vx;
  *vyptr -= vy;
}

__device__ void vectorNormalize(float* vxptr, float* vyptr)
{
  if (!vxptr || !vyptr || isnan(*vxptr) || isnan(*vyptr))
  {
    printf("Cannot normalize invalid vector.\n");
    return;
  }
  float c = vectorGetLength(*vxptr, *vyptr);
  if (c < FLOAT_EPSILON)
  {
    *vxptr = 0;
    *vyptr = 0;
  }
  else
  {
    *vxptr /= c;
    *vyptr /= c;
  }
}

__device__ void vectorInverse(float* vxptr, float* vyptr)
{
  vectorMultiply(vxptr, vyptr, -1);
}

__device__ void vectorMultiply(float* vxptr, float* vyptr, float f)
{
  *vxptr *= f;
  *vyptr *= f;
}

__device__ void vectorCheckValid(float* vxptr, float* vyptr, char* text)
{
  if (!vxptr || !vyptr || isnan(*vxptr) || isnan(*vyptr))
  {
    printf("ERR: %s\n", text);
    return;
  }
}

__device__ float vectorGetLength(float vx, float vy)
{
  if (isnan(vx) || isnan(vy))
  {
    printf("Cannot get length of vector.\n");
    return -1;
  }
  float res = sqrt(vx * vx + vy * vy);
  return res;
}

