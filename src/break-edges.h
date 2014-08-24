
#ifndef BREAK_EDGES_H_
#define BREAK_EDGES_H_

/*!
 * Breaks edges in the graph based on the sign differences in the smoothened
 * values.
 *
 * \param[in] numVertices The number of vertices in the graph.
 * \param[in] fineValues The fine smoothened values.
 * \param[in] coarseValues The coarse smoothened values.
 * \param[in,out] numEdges The number of edges for each vertex.
 * \param[in,out] edgeTargets The targets of the outgoing edges.
 */
void breakEdges(unsigned int numVertices, float* fineValues,
    float* coarseValues, unsigned int* numEdges, unsigned int* edgeTargets);

#endif

