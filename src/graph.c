#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include "graph.h"

Edges* newEdges(unsigned int numvertices)
{
  Edges* edges = (Edges*) calloc(1, sizeof(Edges));
  edges->maxedges = 0;
  edges->numedges = (unsigned int*) calloc(numvertices, sizeof(unsigned int));
  edges->arraySize = numvertices;
  edges->edgeTargets = (unsigned int*) calloc(edges->arraySize,
      sizeof(unsigned int));
  return edges;
}

Vertices* newVertices(unsigned int num)
{
  Vertices* vertices = (Vertices*) calloc(1, sizeof(Vertices));
  vertices->vertexXLocs = (float*) calloc(num, sizeof(float));
  vertices->vertexYLocs = (float*) calloc(num, sizeof(float));
  vertices->numvertices = num;
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
  for (size_t i = 0; i < graph->edges->arraySize; i++)
  {
    graph->edges->edgeTargets[i] = UINT_MAX;
  }
}

void graphAddEdgeToVertex(Graph* graph, unsigned int sourceVertexId,
    unsigned int targetVertexId)
{
  unsigned int index = sourceVertexId
   + graph->edges->numedges[sourceVertexId] * graph->vertices->numvertices;
  if (index >= graph->edges->arraySize)
  {
    graphIncreaseEdgeArraySize(graph, 10);
  }
  graph->edges->numedges[sourceVertexId]++;
  if (graph->edges->numedges[sourceVertexId] > graph->edges->maxedges)
  {
    graph->edges->maxedges = graph->edges->numedges[sourceVertexId];
  }
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

void printGraphEdges(Graph* g)
{
  if (!g)
    return;
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    for (size_t j = 0; j < g->edges->numedges[i]; j++)
    {
      unsigned int index = i + (j * g->vertices->numvertices);
      printf("Edge from %u to %u\n", i, g->edges->edgeTargets[index]);
    }
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

void graphIncreaseEdgeArraySize(Graph* g, unsigned int inc)
{
  unsigned int numelements = g->edges->arraySize
      + (inc * g->vertices->numvertices);
  g->edges->edgeTargets = (unsigned int*) realloc(g->edges->edgeTargets,
      numelements * sizeof(unsigned int));
  if (!g->edges->edgeTargets)
  {
    puts("Could not allocate enough memory for edges.");
    exit(1);
  }
  g->edges->arraySize = numelements;
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    for (size_t j = g->edges->numedges[i]; j < g->edges->maxedges; j++)
    {
      unsigned int index = i + (j * g->vertices->numvertices);
      assert(index < g->edges->arraySize);
      g->edges->edgeTargets[index] = UINT_MAX;
    }
  }
}

void graphShrinkEdgeArrayToActualSize(Graph* g)
{
  unsigned int numelements = g->edges->maxedges * g->vertices->numvertices;
  if (numelements < g->edges->arraySize)
  {
    g->edges->edgeTargets = (unsigned int*) realloc(g->edges->edgeTargets,
        numelements * sizeof(unsigned int));
    g->edges->arraySize = numelements;
  }
}

void graphRandomizeLocation(Graph* g)
{
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    g->vertices->vertexXLocs[i] = rand() % g->vertices->numvertices;
    g->vertices->vertexYLocs[i] = rand() % g->vertices->numvertices;
  }
}
