
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "graph.h"

#define FLOAT_EPSILON 0.0000001

void vectorAdd(double* vxptr, double* vyptr, double vx, double vy)
{
  *vxptr += vx;
  *vyptr += vy;
}

void vectorSubtract(double* vxptr, double* vyptr, double vx, double vy)
{
  *vxptr -= vx;
  *vyptr -= vy;
}

void vectorNormalize(double* vxptr, double* vyptr)
{
  if (!vxptr || !vyptr || isnan(*vxptr) || isnan(*vyptr))
  {
    printf("Cannot normalize invalid vector.\n");
    exit(EXIT_FAILURE);
  }
  double c = vectorGetLength(*vxptr, *vyptr);
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

void vectorInverse(double* vxptr, double* vyptr)
{
  vectorMultiply(vxptr, vyptr, -1);
}

void vectorMultiply(double* vxptr, double* vyptr, double f)
{
  *vxptr *= f;
  *vyptr *= f;
}

void vectorCheckValid(double* vxptr, double* vyptr, char* text)
{
  if (!vxptr || !vyptr || isnan(*vxptr) || isnan(*vyptr))
  {
    printf("ERR: %s\n", text);
    exit(EXIT_FAILURE);
  }
}

double vectorGetLength(double vx, double vy)
{
  if (isnan(vx) || isnan(vy))
  {
    printf("Cannot get length of vector.\n");
    exit(EXIT_FAILURE);
  }
  double res = sqrt(vx * vx + vy * vy);
  return res;
}

