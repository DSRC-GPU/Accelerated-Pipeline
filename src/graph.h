
#ifndef GRAPH_H
#define GRAPH_H

typedef struct Point
{
  // A location in 2D space.
  float x, y;
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
  Vector force, speed;
  // Index in the edges array where the neighbours of this Vertex are located.
  int neighbourIndex, numNeighbours;
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
  unsigned int numedges, numvertices, numneighbours;
  // The array of edges.
  Edge* edges;
  // The array of vertices.
  Vertex* vertices;
} Graph;


// The printGraph function prints the details of the graph to stdout.
void printGraph(Graph*);

// Compares edges based on their start vertex. Used in sorting.
int compare_edges(const void*, const void*);

#endif

