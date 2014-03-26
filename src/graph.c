
#include <string.h>
#include <stdio.h>
#include "graph.h"

void printGraph(Graph* g)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    Vertex* cv = &g->vertices[i];
    printf("Vertex %d\tis at location (%f,%f).\n",
        cv->id, cv->loc.x, cv->loc.y);
  }
}

int compare_edges(const void* e1, const void* e2)
{
  return ((Edge*) e1)->startVertex - ((Edge*) e2)->startVertex;
}

