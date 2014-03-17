
#include <string.h>
#include <stdio.h>

#include "graph.h"

void applyForceOnGraphVertex(Graph* g, void (*func)(Graph*, Vertex*))
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    func(g, &g->vertices[i]);
  }
}

void applyForceOnGraphEdge(Graph* g, void (*func)(Graph*, Edge*))
{
  for (size_t i = 0; i < g->numedges; i++)
  {
    func(g, &g->edges[i]);
  }
}

void getVertexDegrees(Graph* g, unsigned int* degrees)
{
  for (size_t i = 0; i < g->numedges; i++)
  {
    degrees[g->edges[i].start->id]++;
  }
}

void printGraph(Graph* g)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    Vertex* cv = &g->vertices[i];
    printf("Vertex %i is at location (%i,%i) with force vector (%i,%i).\n", 
        cv->id, cv->loc.x, cv->loc.y, cv->force.x, cv->force.y);
  }
  for (size_t i = 0; i < g->numedges; i++)
  {
    Edge* ce = &g->edges[i];
    printf("Vertex %i\tand %i\tare connected by an edge.\n",
        ce->start->id, ce->end->id);
  }
}
