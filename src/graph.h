/*!
 \file graph.h
 A simple graph implementation.
 */
#ifndef GRAPH_H
#define GRAPH_H

/*!
 * Structure that represents a set of edges in a graph.
 */
typedef struct Edges
{
  unsigned int* numedges;
  unsigned int** edgeTargets;
} Edges;

/*!
 * Structure that represents a set of vertices in a graph.
 */
typedef struct Vertices
{
  unsigned int numvertices;
  int* vertexIds;
  float* vertexXLocs;
  float* vertexYLocs;
} Vertices;

/*!
 * Structure that represents a graph.
 */
typedef struct Graph
{
  Edges* edges;
  Vertices* vertices;
} Graph;

/*!
 * Create a new Edges struct with place for the specified number of edges.
 */
Edges* newEdges(unsigned int num);

/*!
 * Allocates memory to save an amount of edges from the specified vertex.
 */
void graphSetEdgeSpaceForVertex(Graph* graph, unsigned int vertexId,
    unsigned int numedges);

/*!
 * Allocates memory to save an amount of edges for all vertices.
 */
void graphSetEdgeSpaceForAllVertices(Graph* graph, unsigned int numedges);

/*!
 * Add an outgoing edge to a vertex. This edge is one-way.
 */
void graphAddEdgeToVertex(Graph* graph, unsigned int sourceVertexId,
    unsigned int targetVertexId);

/*!
 * Reallocate the memory for all edges to match the number of edges each vertex has.
 */
void graphShrinkEdgeSpaceToNumberOfEdges(Graph* graph);

/*!
 * Frees the memory allocated for the edges struct.
 */
void freeEdges(Edges* edges, unsigned int numvertices);

/*!
 * Create a new Vertices struct with place for the specified number of vertices.
 */
Vertices* newVertices(unsigned int num);

/*!
 * Frees the memory allocated for the edges struct.
 */
void freeVertices(Vertices* vertices);

/*!
 * Create a new Graph struct with place for the specified number of edges and vertices.
 */
Graph* newGraph(unsigned int numVertices);

/*!
 Print the details of the graph to stdout.
 \param[in] g The graph to print.
 */
void printGraph(Graph* g);

/*!
 * Free the memory allocated for the graph struct.
 */
void freeGraph(Graph* graph);

#endif

