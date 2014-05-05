
#include <string.h>
#include <stdio.h>
#include "graph.h"

void printGraph(Graph* g)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    printf("%d %.15f %.15f\n",
        g->vertexIds[i] , g->vertexXLocs[i], g->vertexYLocs[i]);
  }
}

