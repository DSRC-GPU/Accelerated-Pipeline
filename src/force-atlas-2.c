
#include <stdlib.h>
#include "force-atlas-2.h"

simpleForce FA2_FORCES[]  = { fa2Gravity, fa2Repulsion, fa2Attraction };

void fa2Gravity(Graph* g, Vertex* v)
{
  // FIXME Implement.
}

void fa2Repulsion(Graph* g, Vertex* v)
{
  // FIXME Implement.
}

void fa2Attraction(Graph* g, Vertex* v)
{
  if (!g || !v) return;
  if (v->neighbourIndex >= 0)
  {
    int index = v->neighbourIndex;
    for (int i = 0; i < v->numNeighbours; i++)
    {
      Edge* e = &g->edges[index + i];
      if (e)
      {
        Vector force;
        getVectorBetweenVertex(v, &g->vertices[e->endVertex], &force);
        addVectors(&v->force, &force);
      }
    }
  }
}

