
#include <stdlib.h>
#include "force-atlas-2.h"

simpleForce FA2_FORCES[]  = { fa2Gravity, fa2Repulsion, fa2Attraction };

void fa2Gravity(Graph* g, Vertex* v)
{
  if (!g || !v) return;
  int k_g = 5;
  int deg = v->numNeighbours;
  Vector gforce = v->loc;
  normalizeVector(&gforce);
  inverseVector(&gforce);
  multiplyVector(&gforce, k_g * (deg + 1));
  addVectors(&v->force, &gforce);
}

void fa2Repulsion(Graph* g, Vertex* v)
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
        normalizeVector(&force);
        inverseVector(&force);

        int deg_n1 = v->numNeighbours + 1;
        int deg_n2 = &g->vertices[e->endVertex].numNeighbours + 1;
        float dist = getVectorLength(&force);

        multiplyVector(&force, K_R * ((deg_n1 + deg_n2) / dist));
        multiplyVector(&force, 0.5);

        addVectors(&v->force, &force);
      }
    }
  }
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
        multiplyVector(&force, 0.5);
        addVectors(&v->force, &force);
      }
    }
  }
}

