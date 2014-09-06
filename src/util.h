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
/*!
 * Void statement.
 */
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

/*!
 * Adds a scalar to a vector.
 *
 * \param[in,out] dst The destination vector.
 * \param[in] scalar The scalar to add.
 * \param[in] num The number of elements in the vector.
 */
void utilVectorAddScalar(float* dst, float scalar, unsigned int num);

/*!
 * Simple gpu vector add in a cuda stream.
 *
 * \param[in,out] dst The destination array.
 * \param[in] src The source array.
 * \param[in] num The number of elements in the arrays.
 * \param[in] stream_ptr The cuda stream to use.
 */
void utilVectorAddInStream(float* dst, float* src, unsigned int num,
    void* stream_ptr);

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

/*!
 * Sums an array.
 *
 * \param[in] d_M The array to sum.
 * \param[in] numelems The number of elements in the array.
 * \param[out] d_outVal pointer to where the output should be written.
 */
void utilParallelSum(float* d_M, unsigned int numelems, float* d_outVal);

/*!
 * Prints an array of floats on the device.
 *
 * \param[in] array The array to print.
 * \param[in] numelems The number of elements to print.
 */
void utilPrintDeviceArray(float* array, unsigned int numelems);

/*!
 * Transfer data from the host to the device. Can free the source of the memory.
 *
 * \param[in] src Pointer to the memory that should be moved.
 * \param[in] numbytes The number of bytes that need to be moved.
 * \param[in] freeHostMem 1 if the src memory should be freed.
 * \return Pointer to the allocated memory.
 */
void* utilDataTransferHostToDevice(void* src, unsigned int numbytes,
    unsigned int freeHostMem);

/*!
 * Transfer data from the device to the host. Can free the source of the memory.
 *
 * \param[in] src Pointer to the memory that should be moved.
 * \param[in] numbytes The number of bytes that need to be moved.
 * \param[in] freeDeviceMem 1 if the src memory should be freed.
 * \return Pointer to the allocated memory.
 */
void* utilDataTransferDeviceToHost(void* src, unsigned int numbytes,
    unsigned int freeDeviceMem);

/*!
 * Allocates memory on the device.
 *
 * \param[in] numbytes The number of bytes to allocate.
 * \return Pointer to the allocated memory.
 */
void* utilAllocateData(unsigned int numbytes);

/*!
 * Frees device memory.
 *
 * \param[in] dptr Pointer to device memory to free.
 */
void utilFreeDeviceData(float* dptr);

/*!
 * Checks and prints cuda errors.
 *
 * \param[in] cudaError_t_ptr Pointer to the error struct.
 * \param[in] msg The message to print if an error is detected.
 */
void utilCudaCheckError(void* cudaError_t_ptr, char* msg);

#endif /* UTIL_H_ */
