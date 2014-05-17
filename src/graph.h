/*!
  \file graph.h
  A simple graph implementation.
 */
#ifndef GRAPH_H
#define GRAPH_H

/*!
  Structure that represents a graph.
 */
typedef struct Graph
{
  // The number of edges and vertices in the graph.
  unsigned int numedges, numvertices;
  int*  vertexIds;
  float* vertexXLocs;
  float* vertexYLocs;
  unsigned int* edgeSources;
  unsigned int* edgeTargets;
} Graph;

/*!
  Print the details of the graph to stdout.
  \param[in] g The graph to print.
 */
void printGraph(Graph* g);

#endif

