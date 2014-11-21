
#include <stdio.h>

Graph* graphParseFile(FILE* ifp)
{
  unsigned int numVertices;
  unsigned int numEdges;
  fscanf(ifp, "%u", &numVertices);
  fscanf(ifp, "%u", &numEdges);

  Graph* graph = newGraph(numVertices);

  for (size_t i = 0, i < numVertices; i++)
  {
    
  }

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
