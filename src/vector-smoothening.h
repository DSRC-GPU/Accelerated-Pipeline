/*
 * \file vector-smoothening.h
 */

#ifndef VECTOR_SMOOTHENING_H_
#define VECTOR_SMOOTHENING_H_

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
    unsigned int numvertices, unsigned int maxedges, unsigned int* edges,
    unsigned int numiterations, float phi);

#endif /* VECTOR_SMOOTHENING_H_ */
