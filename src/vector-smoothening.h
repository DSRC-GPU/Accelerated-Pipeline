/*!
 * \file
 */

#ifndef VECTOR_SMOOTHENING_H_
#define VECTOR_SMOOTHENING_H_

/*!
 *  Prepares memory for the edges on the device and copies the edges to the device.
 *
 *  \param[in] hostEdges The array of edges on the host.
 *  \param[in] hostNumEdges The array of the number of edges per vertex on the host.
 *  \param[in] totaledges The total number of edges.
 *  \param[in] totalvertices The total number of vertices.
 *  \param[out] edges Pointer to the array of the edges on the device.
 *  \param[out] numedges Pointer to the array of edges per vertex on the device.
 */
void vectorSmootheningPrepareEdges(unsigned int* hostEdges,
    unsigned int* hostNumEdges, unsigned int totaledges,
    unsigned int totalvertices, unsigned int** edges, unsigned int** numedges);

/*!
 * Cleans the edge memory that was prepared on the device.
 *
 * \param[in] edges The location of the edges on the device.
 * \param[in] numedges The location of the number of edges per vertex on the device.
 */
void vectorSmootheningCleanEdges(unsigned int* edges, unsigned int* numedges);

/*!
 * Applies vector smoothening.
 *
 * \param[in] xvectors The x-value of the vertex vectors.
 * \param[in] yvectors The y-value of the vertex vectors.
 * \param[in] numvertices The number of vertices.
 * \param[in] maxedges The maximum number of edges per vertex.
 * \param[in] edges The vertex edges.
 * \param[in] numiterations The number of iterations the smoothening
 * should run.
 * \param[in] phi The smoothening constant.
 */
void vectorSmootheningRun(float* xvectors, float* yvectors,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    unsigned int numiterations, float phi);

#endif /* VECTOR_SMOOTHENING_H_ */
