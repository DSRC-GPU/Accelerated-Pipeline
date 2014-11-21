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
  /*!
   * The maximum number of edges for a single vertex.
   */
  unsigned int maxedges;
  /*!
   * An array specifying the number of edges for each vertex.
   */
  unsigned int* numedges;
  /*!
   * An array specifying the target vertex for each edge.
   * This is a 2D array that is column-major mapped to a 1D array.
   * This means every edge for vertex i can be found at:
   * i + numvertices * n
   * where n can range from 0 to maxedges.
   */
  unsigned int* edgeTargets;
  /*!
   * The size of edgeTargets and in number of elements.
   */
  unsigned int arraySize;
} Edges;

/*!
 * Structure that represents a set of vertices in a graph.
 */
typedef struct Vertices
{
  /*!
   * The total number of vertices.
   */
  unsigned int numvertices;
  /*!
   * An array specifying the x location of each vertex.
   */
  float* vertexXLocs;
  /*!
   * An array specifying the y location of each vertex.
   */
  float* vertexYLocs;
} Vertices;

/*!
 * Structure that represents a graph.
 */
typedef struct Graph
{
  /*!
   * The edges in the graph.
   */
  Edges* edges;
  /*!
   * The vertices in the graph.
   */
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
void graphSetEdgeSpaceForAllVertices(Graph* graph);

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
void freeEdges(Edges* edges);

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

void graphShrinkEdgeArrayToActualSize(Graph* g);

#endif

