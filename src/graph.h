
#ifndef GRAPH_H
#define GRAPH_H

typedef struct Point
{
  // A location in 2D space.
  int x, y;
} Point;

typedef Point Vector;

typedef struct Vertex
{
  // The vertex id.
  // A custom label for this vertex.
  // The first time tick this vertex is valid.
  // The last time tick this vertex is valid.
  int id, label, start, end;
  // The location of the vertex in 2D space.
  Point loc;
  // The current force on this vertex.
  Vector force;
} Vertex;

typedef struct Edge
{
  // The array index of the vertex at the start of this edge.
  int startVertex;
  // The array index of the vertex at the end of this edge.
  int endVertex;
  // The first time tick this edge is valid, an the last time tick this edge is
  // valid.
  int start, end;
} Edge;

typedef struct Graph
{
  // The number of edges and vertices in the graph.
  unsigned int numedges, numvertices;
  // The array of edges.
  Edge* edges;
  // The array of vertices.
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

// The printGraph function prints the details of the graph to stdout.
void printGraph(Graph*);

#endif

