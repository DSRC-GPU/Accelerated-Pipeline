#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "graph.h"

#define FLOAT_EPSILON 0.0000001

void vectorAdd(float* vxptr, float* vyptr, float vx, float vy)
{
  *vxptr += vx;
  *vyptr += vy;
}

void vectorSubtract(float* vxptr, float* vyptr, float vx, float vy)
{
  *vxptr -= vx;
  *vyptr -= vy;
}

void vectorNormalize(float* vxptr, float* vyptr)
{
  if (!vxptr || !vyptr || isnan(*vxptr) || isnan(*vyptr))
  {
    printf("Cannot normalize invalid vector.\n");
    exit(EXIT_FAILURE);
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

void vectorInverse(float* vxptr, float* vyptr)
{
  vectorMultiply(vxptr, vyptr, -1);
}

void vectorMultiply(float* vxptr, float* vyptr, float f)
{
  *vxptr *= f;
  *vyptr *= f;
}

void vectorCheckValid(float* vxptr, float* vyptr, char* text)
{
  if (!vxptr || !vyptr || isnan(*vxptr) || isnan(*vyptr))
  {
    printf("ERR: %s\n", text);
    exit(EXIT_FAILURE);
  }
}

float vectorGetLength(float vx, float vy)
{
  if (isnan(vx) || isnan(vy))
  {
    printf("Cannot get length of vector.\n");
    exit(EXIT_FAILURE);
  }
  float res = sqrt(vx * vx + vy * vy);
  return res;
}

