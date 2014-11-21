
#include <stdio.h>

#define BUFFERSIZE 1000

Graph* graphParseFile(FILE* ifp)
{
  char buffer[BUFFERSIZE];
 
  unsigned int numVertices;
  unsigned int numEdges;
  fgets(buffer, BUFFERSIZE, ifp);
  sscanf(buffer, "%u %u", &numVertices, &numEdges);

  Graph* graph = newGraph(numVertices);

  for (size_t i = 0, i < numVertices; i++)
  {
    fgets(buffer, BUFFERSIZE, ifp);
    char* end = buffer;
    unsigned int neighbour = (unsigned int) strol(end, &end, 10);
    // We are making each graph undirected. We do this by ignoring edges that go
    // from (v,u) where v > u. If we find an edge (v,u) where v <= u, we create
    // edges (v,u) and (u,v).
    if (i <= neighbour)
    {
      graphAddEdgeToVertex(i, neighbour);
      graphAddEdgeToVertex(neighbour, i);
    }
  }
  graphShrinkEdgeArrayToActualSize(g);

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
  return graphParseFile(FILE* ifp);    
}
