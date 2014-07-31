#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "graph.h"

Edges* newEdges(unsigned int num)
{
  Edges* edges = (Edges*) calloc(1, sizeof(Edges));
  edges->edgeSources = (unsigned int*) calloc(num, sizeof(float));
  edges->edgeTargets = (unsigned int*) calloc(num, sizeof(float));
  edges->numedges = 0;
  return edges;
}

Vertices* newVertices(unsigned int num)
{
  Vertices* vertices = (Vertices*) calloc(1, sizeof(Vertices));
  vertices->vertexIds = (int*) calloc(num, sizeof(int));
  vertices->vertexXLocs = (float*) calloc(num, sizeof(float));
  vertices->vertexYLocs = (float*) calloc(num, sizeof(float));
  return vertices;
}

Graph* newGraph(unsigned int numedges, unsigned int numvertices)
{
  Graph* graph = (Graph*) calloc(1, sizeof(Graph));
  graph->edges = newEdges(numedges);
  graph->vertices = newVertices(numvertices);
  return graph;
}

void printGraph(Graph* g)
{
  if (!g)
    return;
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    printf("%d %.15f %.15f\n", g->vertices->vertexIds[i],
        g->vertices->vertexXLocs[i], g->vertices->vertexYLocs[i]);
  }
}

