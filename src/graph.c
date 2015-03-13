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
  edges->edgeSet = (unsigned int**) calloc(numvertices, sizeof(unsigned int*));
  edges->numedges = (unsigned int*) calloc(numvertices, sizeof(unsigned int));
  edges->arraySize = numvertices;

  edges->edgeTargets = NULL;
  edges->edgeTargetOffset = NULL;
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

void graphAddEdge(Graph* graph, unsigned int source, unsigned int target)
{
  if (graph)
  {
    unsigned int max = source > target ? source : target;
    max++;

    float* newVertexXLocs;
    float* newVertexYLocs;
    unsigned int* newNumEdges;
    unsigned int** newEdgeSet;
    if (max > graph->vertices->numvertices)
    {
      newVertexXLocs = (float*) realloc(graph->vertices->vertexXLocs, max * sizeof(float));
      newVertexYLocs = (float*) realloc(graph->vertices->vertexYLocs, max * sizeof(float));
      newNumEdges = (unsigned int*) realloc(graph->edges->numedges, max * sizeof(unsigned int));
      newEdgeSet = (unsigned int**) realloc(graph->edges->edgeSet, max * sizeof(unsigned int*));
      if (newVertexXLocs && newVertexYLocs && newNumEdges && newEdgeSet)
      {
        for (size_t i = graph->vertices->numvertices; i < max; i++)
        {
          newVertexXLocs[i] = 0;
          newVertexYLocs[i] = 0;
          newNumEdges[i] = 0;
          newEdgeSet[i] = NULL;
        }

      } else {
        printf("Could not allocate mem for new vertex.\nRequested %u elements.\n", max);
        exit(1);
      }
      graph->vertices->vertexXLocs = newVertexXLocs;
      graph->vertices->vertexYLocs = newVertexYLocs;
      graph->edges->numedges = newNumEdges;
      graph->edges->edgeSet = newEdgeSet;

      graph->vertices->numvertices = max;
      graph->edges->arraySize = max;
    }

    unsigned int numEdgesSource = ++graph->edges->numedges[source];
    unsigned int* newEdgeSetSource = (unsigned int*) realloc(graph->edges->edgeSet[source], numEdgesSource * sizeof(unsigned int));
    if (newEdgeSetSource)
    {
      newEdgeSetSource[numEdgesSource - 1] = target;
      graph->edges->edgeSet[source] = newEdgeSetSource;
    } else {
      puts("Could not allocate mem for new edge.");
    }
  }
}

void graphExportEdges(Graph* graph)
{
  unsigned int totalNumEdges = 0;
  for (size_t i = 0; i < graph->edges->arraySize; i++)
  {
    totalNumEdges += graph->edges->numedges[i];
  }
  unsigned int* edges = (unsigned int*) malloc(totalNumEdges * sizeof(unsigned int));
  unsigned int index = 0;
  for (size_t i = 0; i < graph->edges->arraySize; i++)
  {
    for (size_t j = 0; j < graph->edges->numedges[i]; j++)
    {
      edges[index++] = graph->edges->edgeSet[i][j]; 
    }
  }
  if (graph->edges->edgeTargets)
  {
    free(graph->edges->edgeTargets);
  }
  graph->edges->edgeTargets = edges;
  graph->edges->totalEdges = totalNumEdges;

  unsigned int* edgeTargetOffset = (unsigned int*) calloc(graph->edges->arraySize, sizeof(unsigned int));
  for (size_t i = 1; i < graph->edges->arraySize; i++)
  {
    edgeTargetOffset[i] = edgeTargetOffset[i - 1] + graph->edges->numedges[i - 1];
  }
  if (graph->edges->edgeTargetOffset)
  {
    free(graph->edges->edgeTargetOffset);
  }
  graph->edges->edgeTargetOffset = edgeTargetOffset;
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
      printf("Edge from %lu to %u\n", i, g->edges->edgeSet[i][j]);
    }
  }
}

void freeEdges(Edges* edges)
{
  if (edges)
  {
    if (edges->edgeSet)
    {
      for (size_t i = 0; i < edges->arraySize; i++)
      {
        free(edges->edgeSet[i]);
      }
    }
    free(edges->edgeSet);
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

void graphRandomizeLocation(Graph* g)
{
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    g->vertices->vertexXLocs[i] = rand() % g->vertices->numvertices;
    g->vertices->vertexYLocs[i] = rand() % g->vertices->numvertices;
  }
}
