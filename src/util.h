/*
 * \file util.h
 */

#ifndef UTIL_H_
#define UTIL_H_

#define BLOCK_SIZE 64

void utilVectorAdd(float* dst, float* src, unsigned int num);

void utilVectorAddInStream(float* dst, float* src, unsigned int num,
    cudaStream_t* stream);

void utilVectorMultiply(float* dst, float* src, unsigned int num);

void utilVectorDevide(float* dst, float* src, unsigned int num);

void utilVectorDevideByScalar(float* dst, float denumerator, unsigned int num);

#endif /* UTIL_H_ */
