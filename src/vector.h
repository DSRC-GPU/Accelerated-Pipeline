/*!
  \file vector.h
  Offers basic methods that operate on vectors.
 */
#ifndef VECTOR_H
#define VECTOR_H

#ifdef __CUDACC__
#define DEVICE_FUNC __device__
#else
#define DEVICE_FUNC
#endif

#include "graph.h"

/*!
  Adds the value of vector 2 to vector 1.
  \param vx1 The x value of vector 1.
  \param vy1 The y value of vector 1.
  \param vx2 The x value of vector 2.
  \param vy2 The y value of vector 2.
 */
DEVICE_FUNC void vectorAdd(float* vx1, float* vy1, float vx2, float vy2);

/*! 
  Performs v1 - v2, and stores the result in v1.
  \param vx1 The x value of vector 1.
  \param vy1 The y value of vector 1.
  \param vx2 The x value of vector 2.
  \param vy2 The y value of vector 2.
 */
DEVICE_FUNC void vectorSubtract(float* vx1, float* vy1, float vx2, float vy2);

/*!
  Reduces the length of the given vector to 1.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
 */
DEVICE_FUNC void vectorNormalize(float* vx, float* vy);

/*!
  Convenience method that multiplies the given vector with -1.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
 */
DEVICE_FUNC void vectorInverse(float* vx, float* vy);

/*!
  Multiplies a vector with a scalar and saves the result in the given vector.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
  \param s The scalar with which to multiple the vector.
 */
DEVICE_FUNC void vectorMultiply(float* vx, float* vy, float s);

/*!
  Checks if the vector x and y values are valid numbers (not NaN, not null). If
  this is not the case, prints the given string.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
  \param s The string to print when the vector is invalid.
 */
DEVICE_FUNC void vectorCheckValid(float* vx, float* vy, char* s);

/*!
  Returns the length of the vector.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
 */
DEVICE_FUNC float vectorGetLength(float vx, float vy);

#endif

