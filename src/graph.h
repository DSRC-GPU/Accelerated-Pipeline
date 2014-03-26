
#ifndef GRAPH_H
#define GRAPH_H

typedef struct Graph
{
  // The number of edges and vertices in the graph.
  unsigned int numedges, numvertices;
  float* vertexXLocs;
  float* vertexYLocs;
  unsigned int* edgeSources;
  unsigned int* edgeTargets;
} Graph;


// The printGraph function prints the details of the graph to stdout.
void printGraph(Graph*);

#endif

