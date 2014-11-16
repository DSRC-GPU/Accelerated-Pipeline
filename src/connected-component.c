
#include "connected-component.h"
#include "util.h"
#include <stdlib.h>

inline void min(unsigned int* a, unsigned int b)
{
  *a = (*a < b) ? *a : b;
}

void connectedComponent(unsigned int numvertices, unsigned int* numedges,
    unsigned int* edgeTargets, unsigned int* vertexlabels)
{
  float* f1 = (float*) calloc(numvertices, sizeof(float));
  float* f2 = (float*) calloc(numvertices, sizeof(float));
  utilVectorSetByScalar(f1, 1, numvertices);
  for (size_t i = 0; i < numvertices; i++)
  {
    vertexlabels[i] = i;
  }

  unsigned int m = 1;
  while (m)
  { 
    m = 0;
    for (size_t gid = 0; gid < numvertices; gid++)
    {
      if (f1[gid])
      {
        f1[gid] = 0;
        unsigned int c = vertexlabels[gid];
        unsigned int cmod = 0;
        for (size_t i = 0; i < numedges[gid]; i++)
        {
          unsigned int neighbourIndex = gid + i * numvertices;
          unsigned int neighbour = edgeTargets[neighbourIndex];
          unsigned int cneighbour = vertexlabels[neighbour];
          if (c < cneighbour)
          {
            min(&vertexlabels[neighbour], c);
            f2[neighbour] = 1;
            m = 1;
          }
          else if (c > cneighbour)
          {
            c = cneighbour;
            cmod = 1;
          }
        }
        if (cmod)
        {
          min(&vertexlabels[gid], c);
          f2[gid] = 1;
          m = 1;
        }
      }
    }
  }
  free(f1);
  free(f2);
}

