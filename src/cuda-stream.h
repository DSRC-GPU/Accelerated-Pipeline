/*!
 * \file
 */

#ifndef CUDA_STREAM_H_
#define CUDA_STREAM_H_

/*!
 * Create a new set of streams.
 *
 * \param[in] num The number of streams to create.
 */
cudaStream_t* createCudaStreams(unsigned int num);

/*!
 * Clean up a set of streams.
 *
 * \param[in] streams The array of streams
 * \param[in] num The number of streams in the array.
 */
void cleanCudaStreams(cudaStream_t* streams, unsigned int num);

#endif /* CUDA_STREAM_H_ */
