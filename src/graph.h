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
  unsigned int numedges;
  unsigned int* edgeSources;
  unsigned int* edgeTargets;
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
 * Create a new Vertices struct with place for the specified number of vertices.
 */
Vertices* newVertices(unsigned int num);

/*!
 * Create a new Graph struct with place for the specified number of edges and vertices.
 */
Graph* newGraph(unsigned int numEdges, unsigned int numVertices);

/*!
 Print the details of the graph to stdout.
 \param[in] g The graph to print.
 */
void printGraph(Graph* g);

#endif

