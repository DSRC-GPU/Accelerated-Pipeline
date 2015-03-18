
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "graph.h"

#define BUFFERSIZE 1000

#define FORMAT_FROMTO
//#define FORMAT_LINENUMTO

Graph* graphParseFile(FILE* ifp)
{
  char buffer[BUFFERSIZE];
 
#ifdef FORMAT_LINENUMTO
  unsigned int numVertices;
  unsigned int numEdges;

  sscanf(buffer, "%u %u", &numVertices, &numEdges);

  Graph* graph = newGraph(0);

  for (size_t i = 0; i < numVertices; i++)
  {
    fgets(buffer, BUFFERSIZE, ifp);
    fgets(buffer, BUFFERSIZE, ifp);
    char* end = buffer;
    unsigned int neighbour = 0;
    while((neighbour = (unsigned int) strtol(end, &end, 10)) != 0)
    {
      neighbour--;
      // We are making each graph undirected. We do this by ignoring edges that go
      // from (v,u) where v > u. If we find an edge (v,u) where v <= u, we create
      // edges (v,u) and (u,v).
      if (i <= neighbour)
      {
        graphAddEdge(graph, i, neighbour);
        graphAddEdge(graph, neighbour, i);
      }
    }
  }
#elif defined(FORMAT_FROMTO)
  Graph* graph = newGraph(0);

  unsigned int from, to;
  while (fgets(buffer, BUFFERSIZE, ifp))
  {
    if (buffer[0] != '#')
    {
      sscanf(buffer, "%u\t%u", &from, &to);
      graphAddEdge(graph, from, to);
    }
  }
#else
  puts("No parse format defined.");
  exit(1);
#endif

  return graph;
}

Graph* graphParse(const char* filename)
{
  char* mode = "r";
  FILE* ifp = fopen(filename, mode);

  if (!ifp)
  {
    printf("Could not open file %s.\n", filename);
    exit(1);
  }
  return graphParseFile(ifp);    
}
