#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "graph.h"

Edges* newEdges(unsigned int numvertices)
{
  Edges* edges = (Edges*) calloc(1, sizeof(Edges));
  edges->numedges = (unsigned int*) calloc(numvertices, sizeof(float));
  edges->edgeTargets = (unsigned int**) calloc(numvertices,
      sizeof(unsigned int*));
  return edges;
}

Vertices* newVertices(unsigned int num)
{
  Vertices* vertices = (Vertices*) calloc(1, sizeof(Vertices));
  vertices->vertexIds = (int*) calloc(num, sizeof(int));
  vertices->vertexXLocs = (float*) calloc(num, sizeof(float));
  vertices->vertexYLocs = (float*) calloc(num, sizeof(float));
  vertices->numvertices = 0;
  return vertices;
}

Graph* newGraph(unsigned int numedges, unsigned int numvertices)
{
  Graph* graph = (Graph*) calloc(1, sizeof(Graph));
  graph->edges = newEdges(numedges);
  graph->vertices = newVertices(numvertices);
  return graph;
}

void graphSetEdgeSpaceForVertex(Graph* graph, unsigned int vertexId,
    unsigned int numedges)
{
  graph->edges->edgeTargets[vertexId] = (unsigned int*) calloc(numedges,
      sizeof(unsigned int));
}

void graphSetEdgeSpaceForAllVertices(Graph* graph, unsigned int numedges)
{
  for (size_t i = 0; i < graph->vertices->numvertices; i++)
  {
    graph->edges->edgeTargets[i] = (unsigned int*) calloc(numedges,
        sizeof(unsigned int));
  }
}

void graphAddEdgeToVertex(Graph* graph, unsigned int sourceVertexId,
    unsigned int targetVertexId)
{
  unsigned int index = graph->edges->numedges[sourceVertexId]++;
  graph->edges->edgeTargets[sourceVertexId][index - 1] = targetVertexId;
}

void graphShrinkEdgeSpaceToNumberOfEdges(Graph* graph)
{
  for (size_t i = 0; i < graph->vertices->numvertices; i++)
  {
    graph->edges->edgeTargets[i] = (unsigned int*) realloc(
        graph->edges->edgeTargets[i], graph->edges->numedges[i]);
  }
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

void freeEdges(Edges* edges, unsigned int numvertices)
{
  for (size_t i = 0; i < numvertices; i++)
  {
    free(edges->edgeTargets[i]);
  }
  free(edges->edgeTargets);
  free(edges->numedges);
  free(edges);
}

void freeVertices(Vertices* vertices)
{
  free(vertices->vertexIds);
  free(vertices->vertexXLocs);
  free(vertices->vertexYLocs);
  free(vertices);
}

void freeGraph(Graph* graph)
{
  freeEdges(graph->edges, graph->vertices->numvertices);
  freeVertices(graph->vertices);
  free(graph);
}
