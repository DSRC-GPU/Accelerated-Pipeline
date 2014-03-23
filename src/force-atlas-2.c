
#include <stdlib.h>
#include "force-atlas-2.h"
#include "math.h"

#define FA2_NUMFORCES 3
#define K_R 1.0
#define K_S 0.1
#define K_SMAX 10.0
#define TAU 1.0

typedef struct VertexData
{
  float tra, swg, speed;
  Vector oldForce;
} VertexData;

// Gravity force
void fa2Gravity(Graph*, Vertex*);
// Repulsion between vertices
void fa2Repulsion(Graph*, Vertex*);
// Attraction on edges
void fa2Attraction(Graph*, Vertex*);

// Array of forces.
simpleForce FA2_FORCES[]  = { fa2Gravity, fa2Repulsion, fa2Attraction };

void fa2UpdateSwing(Graph*, VertexData*);
void fa2UpdateTract(Graph*, VertexData*);
void fa2UpdateSwingGraph(Graph*, VertexData*, float*);
void fa2UpdateTractGraph(Graph*, VertexData*, float*);
void fa2UpdateSpeedGraph(float, float, float*);
void fa2UpdateSpeed(Graph*, VertexData*, float);
void fa2UpdateDisplacement(Graph*, VertexData*);
void fa2SaveOldForces(Graph*, VertexData*);

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
        int deg_n2 = g->vertices[e->endVertex].numNeighbours + 1;
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

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
void fa2UpdateSwing(Graph* g, VertexData* vd)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    Vector v = g->vertices[i].force;
    subtractVectors(&v, &vd[i].oldForce);
    float vlen = getVectorLength(&v);
    vd[i].swg = vlen;
  }
}

// Updates the traction for each vertex, as described in the Force Atlas 2
// paper.
void fa2UpdateTract(Graph* g, VertexData* vd)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    Vector v = g->vertices[i].force;
    addVectors(&v, &vd[i].oldForce);
    float vlen = getVectorLength(&v);
    vd[i].swg = vlen / 2;
  }
}

// Calculate the current swing of the graph.
void fa2UpdateSwingGraph(Graph* g, VertexData* vd, float* gswing)
{
  *gswing = 0;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    *gswing += (g->vertices[i].numNeighbours + 1) * vd[i].swg;
  }
}

// Calculate the current traction of the graph.
void fa2UpdateTractGraph(Graph* g, VertexData* vd, float* gtract)
{
  *gtract = 0;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    *gtract += (g->vertices[i].numNeighbours + 1) * vd[i].tra;
  }
}

void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed)
{
  *gspeed = TAU * (gtract / gswing);
}

void fa2UpdateSpeed(Graph* g, VertexData* vd, float gs)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    vd[i].speed = K_S * gs / (1 + (gs * sqrt(vd[i].swg)));
    vd[i].speed = fmin(vd[i].speed,
        K_SMAX / getVectorLength(&g->vertices[i].force));
  }
}

// Save current forces as the previous forces for the next tick.
void fa2SaveOldForces(Graph* g, VertexData* vd)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    vd[i].oldForce = g->vertices[i].force;
  }
}

void fa2UpdateDisplacement(Graph* g, VertexData* vd)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    g->vertices[i].displacement = g->vertices[i].force;
    multiplyVector(&g->vertices[i].displacement, vd[i].speed);
  }
}

void fa2RunOnGraph(Graph* g)
{
  static VertexData* vdata = NULL;
  static float graphSwing = 0.0;
  static float graphTract = 0.0;
  static float graphSpeed = 0.0;

  if (!vdata)
    vdata = calloc(g->numvertices, sizeof(VertexData)); 

  // Compute forces.
  updateForcesOnGraph(g, FA2_NUMFORCES, FA2_FORCES);

  // Calculate speed of vertices.
  // Update swing of vertices.
  fa2UpdateSwing(g, vdata);

  // Update traction of vertices.
  fa2UpdateTract(g, vdata);

  // Update swing of Graph.
  fa2UpdateSwingGraph(g, vdata, &graphSwing);

  // Update traction of Graph.
  fa2UpdateTractGraph(g, vdata, &graphTract);

  // Update speed of Graph.
  fa2UpdateSpeedGraph(graphSwing, graphTract, &graphSpeed);

  // Update speed of vertices.
  fa2UpdateSpeed(g, vdata, graphSpeed);

  // Update displacement of vertices.
  fa2UpdateDisplacement(g, vdata);

  // Set current forces as old forces in vertex data.
  fa2SaveOldForces(g, vdata);

  // Update vertex locations based on speed.
  updateLocationOnGraph(g);

  // Reset forces on vertices to 0.
  resetForcesOnGraph(g);
}

