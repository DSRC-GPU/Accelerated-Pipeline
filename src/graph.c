
#include <string.h>
#include <stdio.h>
#include "graph.h"

void printGraph(Graph* g)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    Vertex* cv = &g->vertices[i];
    printf("Vertex %d\tis at location (%f,%f)\t\
        with displacement vector (%f,%f).\n",
        cv->id, cv->loc.x, cv->loc.y, cv->displacement.x, cv->displacement.y);
    printf("Vertex has %d neighbours, starting at %d.\n", cv->numNeighbours,
        cv->neighbourIndex);
    
    // DEBUG break.
    break;
  }
  for (size_t i = 0; i < g->numedges; i++)
  {
    // DEBUG break.
    break;

    Edge* ce = &g->edges[i];
    printf("Vertex %i\tand %i\tare connected by an edge.\n",
        ce->startVertex, ce->endVertex);
  }
}

int compare_edges(const void* e1, const void* e2)
{
  return ((Edge*) e1)->startVertex - ((Edge*) e2)->startVertex;
}

