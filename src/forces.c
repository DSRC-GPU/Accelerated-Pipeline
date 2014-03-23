
#include <stdlib.h>
#include <math.h>
#include "forces.h"
#include "graph.h"

void applyForceOnGraph(Graph*, simpleForce);
void applyDataForceOnGraph(Graph*, ForceWithData*);

// Give a set of forces that are to be used by the spring embedding core.
void runForcesOnGraph(Graph*, unsigned int, simpleForce*);
void runDataForcesOnGraph(Graph*, unsigned int, ForceWithData*);

// FIXME sent tick num, to ignore disabled vertices.
void applyForceOnGraph(Graph* g, simpleForce sf)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    sf(g, &g->vertices[i]);
  }
}

// FIXME sent tick num, to ignore disabled vertices.
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

void updateLocationOnGraph(Graph* g)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    addVectors((Vector*) &g->vertices[i].loc, &g->vertices[i].displacement);
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
  *vec = v2->loc;
  subtractVectors(vec, (Vector*) &v1->loc);
}

float getVectorLength(Vector* v)
{
  return sqrt(v->x * v->x + v->y * v->y);
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

void normalizeVector(Vector* v)
{
  float c = getVectorLength(v);
  v->x /= c;
  v->y /= c;
}

void inverseVector(Vector* v)
{
  multiplyVector(v, -1);
}

void multiplyVector(Vector* v, float f)
{
  v->x *= f;
  v->y *= f;
}

