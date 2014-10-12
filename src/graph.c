#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include "graph.h"

Edges* newEdges(unsigned int numvertices)
{
  Edges* edges = (Edges*) calloc(1, sizeof(Edges));
  edges->numedges = (unsigned int*) calloc(numvertices, sizeof(unsigned int));
  edges->edgeTargets = (unsigned int*) calloc(numvertices,
      sizeof(unsigned int));
  return edges;
}

Vertices* newVertices(unsigned int num)
{
  Vertices* vertices = (Vertices*) calloc(1, sizeof(Vertices));
  vertices->vertexXLocs = (float*) calloc(num, sizeof(float));
  vertices->vertexYLocs = (float*) calloc(num, sizeof(float));
  vertices->numvertices = 0;
  return vertices;
}

Graph* newGraph(unsigned int numvertices)
{
  Graph* graph = (Graph*) calloc(1, sizeof(Graph));
  graph->edges = newEdges(numvertices);
  graph->vertices = newVertices(numvertices);
  return graph;
}

void graphSetEdgeSpaceForAllVertices(Graph* graph)
{
  unsigned int* targets = graph->edges->edgeTargets;
  if (targets)
    free(targets);
  graph->edges->edgeTargets = (unsigned int*) calloc(
      graph->vertices->numvertices * graph->edges->maxedges,
      sizeof(unsigned int));
  for (size_t i = 0; i < graph->vertices->numvertices * graph->edges->maxedges;
      i++)
  {
    graph->edges->edgeTargets[i] = UINT_MAX;
  }
}

void graphAddEdgeToVertex(Graph* graph, unsigned int sourceVertexId,
    unsigned int targetVertexId)
{
  unsigned int index = sourceVertexId
      + graph->edges->numedges[sourceVertexId]++ * graph->vertices->numvertices;
  graph->edges->edgeTargets[index] = targetVertexId;
}

void printGraph(Graph* g)
{
  if (!g)
    return;
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    printf("%zu %.15f %.15f\n", i,
        g->vertices->vertexXLocs[i], g->vertices->vertexYLocs[i]);
  }
}

void freeEdges(Edges* edges)
{
  if (edges)
  {
    free(edges->edgeTargets);
    free(edges->numedges);
    free(edges);
  }
}

void freeVertices(Vertices* vertices)
{
  if (vertices)
  {
    free(vertices->vertexXLocs);
    free(vertices->vertexYLocs);
    free(vertices);
  }
}

void freeGraph(Graph* graph)
{
  freeEdges(graph->edges);
  freeVertices(graph->vertices);
  free(graph);
}
