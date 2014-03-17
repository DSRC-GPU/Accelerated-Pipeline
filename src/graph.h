
#ifndef GRAPH_H
#define GRAPH_H

typedef struct Point
{
  int x, y;
} Point;

typedef Point Vector;

typedef struct Vertex
{
  int id;
  Point loc;
  Vector force;
} Vertex;

typedef struct Edge
{
  Vertex* start;
  Vertex* end;
} Edge;

typedef struct Graph
{
  unsigned int numedges, numvertices;
  Edge* edges;
  Vertex* vertices;
} Graph;

// The applyForceOnGraphVertex function is a convenience function that iterates
// over all vertices in the graph. This allows the user to update Vertex forces
// based on vertex information.
void applyForceOnGraphVertex(Graph*, void (*func)(Graph*, Vertex*));

// The applyForceOnGraphEdge function is a convenience function that iterates
// over all edges in the graph. This allows the user to update Vertex forces
// based on edge information.
void applyForceOnGraphEdge(Graph*, void (*func)(Graph*, Edge*));

// The getVertexDegrees function calculates the degree of each vertex and writes
// it to the given unsigned int array. The size of the given array should be
// equal to the number of vertices in the graph.
void getVertexDegrees(Graph*, unsigned int*);

// The printGraph function prints the details of the graph to stdout.
void printGraph(Graph*);

#endif
