
#include "break-edges.h"
#include <stdlib.h>

inline int sgnCmp(float a, float b)
{
  return a * b >= 0;
}

void breakEdges(unsigned int numVertices, float* fineValues,
    float* coarseValues, unsigned int* numEdges, unsigned int* edgeTargets)
{
  for (size_t gid = 0; gid < numVertices; gid++)
  {
    unsigned int localEdges = numEdges[gid];
    float localValue = fineValues[gid] - coarseValues[gid];
    for (size_t i = 0; i < localEdges; i++)
    {
      unsigned int index = numVertices * i + gid;
      unsigned int neighbour = edgeTargets[index];
      float neighbourValue = fineValues[neighbour] - coarseValues[neighbour];
      if (!sgnCmp(localValue, neighbourValue))
      {
        // Removing edge by setting target to itself.
        edgeTargets[index] = gid;
      }
    }
  }
}

