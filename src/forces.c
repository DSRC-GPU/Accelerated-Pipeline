
#include <stdlib.h>
#include "forces.h"
#include "graph.h"

void applyForceOnGraph(Graph*, simpleForce);
void applyDataForceOnGraph(Graph*, ForceWithData*);

// Give a set of forces that are to be used by the spring embedding core.
void runForcesOnGraph(Graph*, unsigned int, simpleForce*);
void runDataForcesOnGraph(Graph*, unsigned int, ForceWithData*);

void applyForceOnGraph(Graph* g, simpleForce sf)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    sf(g, &g->vertices[i]);
  }
}

void applyDataForceOnGraph(Graph* g, ForceWithData* fwd)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    fwd->force(g, &g->vertices[i], fwd->data);
  }
}

// Give a set of forces that are to be used by the spring embedding core.
void updateForcesOnGraph(Graph* g, unsigned int num, simpleForce* sf)
{
  for (size_t i = 0; i < num; i++)
  {
    applyForceOnGraph(g, sf[i]);
  }
}

void updateDataForcesOnGraph(Graph* g, unsigned int num, ForceWithData* fwd)
{
  for (size_t i = 0; i < num; i++)
  {
    applyDataForceOnGraph(g, &fwd[i]);
  }
}

void updateSpeedOnGraph(Graph* g)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    addVectors(&g->vertices[i].speed, &g->vertices[i].force);
  }
}

void updateLocationOnGraph(Graph* g)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    addVectors((Vector*) &g->vertices[i].loc, &g->vertices[i].speed);
  }
}

void resetForcesOnGraph(Graph* g)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    g->vertices[i].force.x = 0;
    g->vertices[i].force.y = 0;
  }
}

void getVectorBetweenVertex(Vertex* v1, Vertex* v2, Vector* vec)
{
  *vec = (Vector) v2->loc;
  subtractVectors(vec, (Vector*) &v1->loc);
}

void addVectors(Vector* v1, Vector* v2)
{
  v1->x += v2->x;
  v1->y += v2->y;
}

void subtractVectors(Vector* v1, Vector* v2)
{
  v1->x -= v2->x;
  v1->y -= v2->y;
}

