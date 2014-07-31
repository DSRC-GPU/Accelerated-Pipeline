/*
 * \file speedvector.h
 */

#ifndef SPEEDVECTOR_H_
#define SPEEDVECTOR_H_

/*!
 * Calculates the speed vectors for all vertices for a particular window size.
 *
 * \param[in] windowsize The window size that is used. Should equal the length
 * of the verticesArray and the vectors array.
 * \param[in] verticesArray An array of Vertices that provide vertex locations
 * for each point in time within the current window.
 * \param[out] The speed vectors for each vertex. X location is stored in 0, Y
 * location is stored in 1.
 */
void calculateSpeedVectors(unsigned int windowsize, Vertices** verticesArray,
    float** vectors);

#endif /* SPEEDVECTOR_H_ */
