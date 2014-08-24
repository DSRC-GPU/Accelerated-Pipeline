
#include <stdlib.h>
#include <string.h>

void smootheningPrepareEdges(unsigned int* hostEdges,
    unsigned int* hostNumEdges, unsigned int totaledges,
    unsigned int totalvertices, unsigned int** edges, unsigned int** numedges)
{
  *edges = hostEdges;
  *numedges = hostNumEdges;
}

void smootheningPrepareOutput(float** output, unsigned int numvertices)
{
  *output = (float*) calloc(numvertices, sizeof(float));
}

void smootheningCleanEdges(unsigned int* edges, unsigned int* numedges)
{
  free(edges);
  free(numedges);
}

void smootheningRun(float* values,
    unsigned int numvertices, unsigned int* numedges, unsigned int* edges,
    unsigned int numiterations, float phi, float* smoothValues)
{
  memcpy(smoothValues, values, numvertices * sizeof(float));

  float* res = (float*) calloc(numvertices, sizeof(float));
  for (size_t i = 0; i < numiterations; i++)
  {
    for (size_t gid = 0; gid < numvertices; gid++)
    {
      res[gid] = phi * values[gid];
      for (size_t i = 0; i < numedges[gid]; i++)
      {
        unsigned int index = edges[gid + (numvertices * i)];
        res[gid] += ((1 - phi) * smoothValues[index]) / numedges[gid];
      }
    }
    for (size_t gid = 0; gid < numvertices; gid++)
      smoothValues[gid] = res[gid];
  }
  free(res);
}

