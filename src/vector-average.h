/*!
 * \file
 */

#ifndef VECTOR_AVERAGE_H_
#define VECTOR_AVERAGE_H_

/*!
 * The number of spring embedding runs that will be used to compute the average
 * vectors.
 */
#define WINDOW_SIZE 10

/*!
 * Allocates a new array on the device to store the speed vectors.
 *
 * \param[in] numelements The number of vectors that need to be stored. The
 * vector is in 2D and will be saved in an row major format, resulting in an
 * 2*numelements sized array.
 * \return An new array on the device.
 */
float* vectorAverageNewVectorArray(unsigned int numelements);

/*!
 * Frees an array on the device.
 *
 * \param[in] averageArray the array on the device to free.
 */
void vectorAverageFreeVectorArray(float* averageArray);

/*!
 *  Creates a new window. This is an array of size WINDOW_SIZE and is allocated
 *  on the host. The enties of the window should point to arrays on the device.
 */
float** vectorAverageNewWindow();

/*!
 *  Frees the allocated window on the host and the average arrays on the device
 *  that the window points to.
 */
void vectorAverageFreeWindow(float** window);

/*!
 * Shifts the entries in the window. Frees the memory on the device for the
 * entry that is removed in the shift. Places the new entry in the free spot.
 */
void vectorAverageShiftAndAdd(float** window, float* newEntry);

/*!
 * Compute the average speed vectors from the current window.
 */
void vectorAverageComputeAverage(float** window, unsigned int numelements,
    float* average);

#endif
