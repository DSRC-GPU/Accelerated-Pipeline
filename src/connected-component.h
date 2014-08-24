
#ifndef CONNECTED_COMPONENT_H_
#define CONNECTED_COMPONENT_H_

/*!
 * Computes the connected components among the vertices based on the given
 * vertices.
 *
 * \param[in] numvertices The number of vertices in the graph.
 * \param[in] numedges The number of edges for each vertex.
 * \param[in] edgeTargets The edges of the graph.
 * \param[out] vertexlabels The labels for the edges. Each connected component
 * has one label.
 */
void connectedComponent(unsigned int numvertices, unsigned int* numedges,
      unsigned int* edgeTargets, unsigned int* vertexlabels);

#endif

