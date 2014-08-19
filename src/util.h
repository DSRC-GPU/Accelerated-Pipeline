/*!
 * \file
 */

#ifndef UTIL_H_
#define UTIL_H_

/*!
 * Prints a statement with printf as long as 'DEBUG' is defined while compiling.
 */
#ifdef DEBUG
#define DEBUG_PRINT printf
#define DEBUG_PRINT_DEVICE utilPrintDeviceArray
#else
#define DEBUG_PRINT(...)
#define DEBUG_PRINT_DEVICE(...)
#endif

#include <cuda_runtime.h>

/*!
 * Default block size of all cuda kernels.
 */
#define BLOCK_SIZE 64

/*!
 * Sets all values in an array to a scalar value.
 *
 * \param[in,out] dst The array whose values should be set.
 * \param[in] scalar The scaler value to use.
 * \param[in] num The number of elements to set.
 */
void utilVectorSetByScalar(float* dst, float scalar, unsigned int num);

/*!
 * Simple gpu vector add.
 *
 * \param[in,out] dst The destination array.
 * \param[in] src The source array.
 * \param[in] num The number of elements in the arrays.
 */
void utilVectorAdd(float* dst, float* src, unsigned int num);

void utilVectorAddScalar(float* dst, float scalar, unsigned int num);

/*!
 * Simple gpu vector add in a cuda stream.
 *
 * \param[in,out] dst The destination array.
 * \param[in] src The source array.
 * \param[in] num The number of elements in the arrays.
 * \param[in] stream The cuda stream to use.
 */
void utilVectorAddInStream(float* dst, float* src, unsigned int num,
    cudaStream_t* stream);

/*!
 * Simple gpu vector multiply.
 *
 * \param[in,out] dst The destination array.
 * \param[in] src The source array.
 * \param[in] num The number of elements in the arrays.
 */
void utilVectorMultiply(float* dst, float* src, unsigned int num);

/*!
 * Multiple all elements in an array by a scalar.
 *
 * \param[in,out] dst The destination array.
 * \param[in] scalar The scalar to use in the multiplication.
 * \param[in] num The number of elements in the array.
 */
void utilVectorMultiplyByScalar(float* dst, float scalar, unsigned int num);

/*!
 * Simple gpu vector device.
 *
 * \param[in,out] dst The destination array.
 * \param[in] src The source array.
 * \param[in] num The number of elements in the arrays.
 */
void utilVectorDevide(float* dst, float* src, unsigned int num);

/*!
 * Devide all elements in an array by a scalar.
 *
 * \param[in,out] dst The destination array.
 * \param[in] scalar The scalar to use in the devision.
 * \param[in] num The number of elements in the array.
 */
void utilVectorDevideByScalar(float* dst, float scalar, unsigned int num);

void utilTreeReduction(float* d_M, unsigned int numelems, float* d_outVal);

void utilPrintDeviceArray(float* array, unsigned int numelems);

float* utilDataTransferHostToDevice(float* src, unsigned int
    numbytes, unsigned int freeHostMem);

void utilFreeDeviceData(float* dptr);

#endif /* UTIL_H_ */
