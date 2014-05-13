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
DEVICE_FUNC void vectorAdd(double* vx1, double* vy1, double vx2, double vy2);

/*! 
  Performs v1 - v2, and stores the result in v1.
  \param vx1 The x value of vector 1.
  \param vy1 The y value of vector 1.
  \param vx2 The x value of vector 2.
  \param vy2 The y value of vector 2.
 */
DEVICE_FUNC void vectorSubtract(double* vx1, double* vy1, double vx2, double vy2);

/*!
  Reduces the length of the given vector to 1.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
 */
DEVICE_FUNC void vectorNormalize(double* vx, double* vy);

/*!
  Convenience method that multiplies the given vector with -1.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
 */
DEVICE_FUNC void vectorInverse(double* vx, double* vy);

/*!
  Multiplies a vector with a scalar and saves the result in the given vector.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
  \param s The scalar with which to multiple the vector.
 */
DEVICE_FUNC void vectorMultiply(double* vx, double* vy, double s);

/*!
  Checks if the vector x and y values are valid numbers (not NaN, not null). If
  this is not the case, prints the given string.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
  \param s The string to print when the vector is invalid.
 */
DEVICE_FUNC void vectorCheckValid(double* vx, double* vy, char* s);

/*!
  Returns the length of the vector.
  \param vx The x value of the vector.
  \param vy The y value of the vector.
 */
DEVICE_FUNC double vectorGetLength(double vx, double vy);

#endif

