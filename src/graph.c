
#include <string.h>
#include <stdio.h>
#include "graph.h"

void printGraph(Graph* g)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    Vertex* cv = &g->vertices[i];
    printf("Vertex %i\tis at location (%i,%i)\twith force vector (%i,%i).\n", 
        cv->id, cv->loc.x, cv->loc.y, cv->force.x, cv->force.y);
  }
  for (size_t i = 0; i < g->numedges; i++)
  {
    Edge* ce = &g->edges[i];
    printf("Vertex %i\tand %i\tare connected by an edge.\n",
        ce->start, ce->end);
  }
}

