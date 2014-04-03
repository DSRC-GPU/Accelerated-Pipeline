
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "force-atlas-2.h"
#include "math.h"
#include "vector.h"

#define K_R 1.0
#define K_S 0.1
#define K_G 5
#define K_SMAX 10.0
#define TAU 1.0
#define EPSILON 0.1
#define FLOAT_EPSILON 0.0000001

// Calculate the number of neighbours for every node.
void calcNumNeighbours(Graph*, int*);
// Gravity force
void fa2Gravity(Graph*, float*, float*, int*);
// Repulsion between vertices
void fa2Repulsion(Graph*, float*, float*, int*);
// Attraction on edges
void fa2Attraction(Graph*, float*, float*);

void fa2UpdateSwing(Graph*, float*, float*, float*, float*, float*);
void fa2UpdateTract(Graph*, float*, float*, float*, float*, float*);
void fa2UpdateSwingGraph(Graph*, float*, int*, float*);
void fa2UpdateTractGraph(Graph*, float*, int*, float*);
void fa2UpdateSpeedGraph(float, float, float*);
void fa2UpdateSpeed(Graph*, float*, float*, float*, float*, float);
void fa2UpdateDisplacement(Graph*, float*, float*, float*, float*, float*);
void fa2SaveOldForces(Graph*, float*, float*, float*, float*);
void fa2UpdateLocation(Graph*, float*, float*);

void calcNumNeighbours(Graph* g, int* deg)
{
  for (size_t i = 0; i < g->numedges; i++)
  {
    deg[g->edgeSources[i]]++; 
  }
}

void fa2Gravity(Graph* g, float* forceX, float* forceY, int* deg)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    float vx = g->vertexXLocs[i];
    float vy = g->vertexYLocs[i];
    vectorNormalize(&vx, &vy);
    vectorInverse(&vx, &vy);
    vectorMultiply(&vx, &vy, K_G * (deg[i] + 1));
    vectorAdd(&forceX[i], &forceY[i], vx, vy);
  }
}

void fa2Repulsion(Graph* g, float* forceX, float* forceY, int* deg)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    for (size_t j = 0; j < g->numvertices; j++)
    {
      if (i == j) continue;
      float vx1 = g->vertexXLocs[i];
      float vy1 = g->vertexYLocs[i];
      float vx2 = g->vertexXLocs[j];
      float vy2 = g->vertexYLocs[j];

      vectorSubtract(&vx1, &vy1, vx2, vy2);
      float dist = vectorGetLength(vx1, vy1);
      if (dist < FLOAT_EPSILON)
        dist = EPSILON;

      vectorNormalize(&vx1, &vy1);
      vectorMultiply(&vx1, &vy1, K_R * (((deg[i] + 1) * (deg[j] + 1))
              / dist));
      vectorMultiply(&vx1, &vy1, 0.5);

      vectorAdd(&forceX[i], &forceY[i], vx1, vy1);
    }
  }
}

void fa2Attraction(Graph* g, float* forceX, float* forceY)
{
  if (!g) return;
  for (size_t i = 0; i < g->numedges; i++)
  {
    int v1Index = g->edgeSources[i];
    int v2Index = g->edgeTargets[i];

    float vx1 = g->vertexXLocs[v1Index];
    float vy1 = g->vertexYLocs[v1Index];
    float vx2 = g->vertexXLocs[v2Index];
    float vy2 = g->vertexYLocs[v2Index];

    vectorSubtract(&vx2, &vy2, vx1, vy1);
    vectorMultiply(&vx2, &vy2, 0.5);
    vectorAdd(&forceX[v1Index], &forceY[v1Index], vx2, vy2);
  }
}

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
void fa2UpdateSwing(Graph* g, float* forceX, float* forceY,
    float* oldForceX, float* oldForceY, float* swg)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    float fx = forceX[i];
    float fy = forceY[i];
    vectorSubtract(&fx, &fy, oldForceX[i], oldForceY[i]);
    float vlen = vectorGetLength(fx, fy);
    swg[i] = vlen;
  }
}

// Updates the traction for each vertex, as described in the Force Atlas 2
// paper.
void fa2UpdateTract(Graph* g, float* forceX, float* forceY,
    float* oldForceX, float* oldForceY, float* tra)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    float fx = forceX[i];
    float fy = forceY[i];
    vectorAdd(&fx, &fy, oldForceX[i], oldForceY[i]);
    float vlen = vectorGetLength(fx, fy);
    tra[i] = vlen / 2;
  }
}

// Calculate the current swing of the graph.
void fa2UpdateSwingGraph(Graph* g, float* swg, int* deg, float* gswing)
{
  *gswing = 0;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    *gswing += (deg[i] + 1) * swg[i];
  }
}

// Calculate the current traction of the graph.
void fa2UpdateTractGraph(Graph* g, float* tra, int* deg, float* gtract)
{
  *gtract = 0;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    *gtract += (deg[i] + 1) * tra[i];
  }
}

void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed)
{
  float oldSpeed = *gspeed;
  *gspeed = gswing > 0 ? TAU * (gtract / gswing) : EPSILON;
  if (*gspeed <= 0)
    *gspeed = EPSILON;
  // Do not allow more then 50% speed increase.
  if (oldSpeed > FLOAT_EPSILON && *gspeed > 1.5 * oldSpeed)
    *gspeed = 1.5 * oldSpeed;
}

void fa2UpdateSpeed(Graph* g, float* speed, float* swg, float* forceX,
    float* forceY, float gs)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    float vSwg = swg[i];
    if (vSwg <= 0)
      vSwg = EPSILON;
    float vForceLen = vectorGetLength(forceX[i], forceY[i]);
    if (vForceLen <= 0)
      vForceLen = EPSILON;

    speed[i] = K_S * gs / (1 + (gs * sqrt(vSwg)));
    speed[i] = fmin(speed[i],
        K_SMAX / vForceLen);
  }
}

// Save current forces as the previous forces for the next tick.
void fa2SaveOldForces(Graph* g, float* forceX, float* forceY,
    float* oldForceX, float* oldForceY)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    oldForceX[i] = forceX[i];
    oldForceY[i] = forceY[i];
  }
}

void fa2UpdateDisplacement(Graph* g, float* speed, float* forceX,
    float* forceY, float* dispX, float* dispY)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    dispX[i] = forceX[i];
    dispY[i] = forceY[i];
    vectorMultiply(&dispX[i], &dispY[i], speed[i]);
  }
}

void fa2UpdateLocation(Graph* g, float* xdisp, float* ydisp)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    g->vertexXLocs[i] += xdisp[i];
    g->vertexYLocs[i] += ydisp[i];
  }
}

void fa2RunOnce(Graph* g)
{
  static int firstRun = 1;
  static int* numNeighbours = NULL;
  static float* tra = NULL;
  static float* swg = NULL;
  static float* speed = NULL;
  static float* forceX = NULL;
  static float* forceY = NULL;
  static float* oldForceX = NULL;
  static float* oldForceY = NULL;
  static float* dispX = NULL;
  static float* dispY = NULL;

  float graphSwing = 0.0;
  float graphTract = 0.0;
  float graphSpeed = 0.0;

  if (firstRun)
  {
    numNeighbours = (int*) calloc(g->numvertices, sizeof(int));
    calcNumNeighbours(g, numNeighbours);
    tra = (float*) calloc(g->numvertices, sizeof(float));
    swg = (float*) calloc(g->numvertices, sizeof(float));
    speed = (float*) calloc(g->numvertices, sizeof(float));
    forceX = (float*) calloc(g->numvertices, sizeof(float));
    forceY = (float*) calloc(g->numvertices, sizeof(float));
    oldForceX = (float*) calloc(g->numvertices, sizeof(float));
    oldForceY = (float*) calloc(g->numvertices, sizeof(float));
    dispX = (float*) calloc(g->numvertices, sizeof(float));
    dispY = (float*) calloc(g->numvertices, sizeof(float));

    firstRun = 0;
  }

  // Update swing of Graph.
  fa2UpdateSwingGraph(g, swg, numNeighbours, &graphSwing);

  // Update traction of Graph.
  fa2UpdateTractGraph(g, tra, numNeighbours, &graphTract);

  // Update speed of Graph.
  fa2UpdateSpeedGraph(graphSwing, graphTract, &graphSpeed);

  // Reset forces on vertices to 0.
  memset(forceX, 0, sizeof(float) * g->numvertices);
  memset(forceY, 0, sizeof(float) * g->numvertices);

  // Gravity force
  fa2Gravity(g, forceX, forceY, numNeighbours);
  // Repulsion between vertices
  fa2Repulsion(g, forceX, forceY, numNeighbours);
  // Attraction on edges
  fa2Attraction(g, forceX, forceY);

  // Calculate speed of vertices.
  // Update swing of vertices.
  fa2UpdateSwing(g, forceX, forceY, oldForceX, oldForceY, swg);

  // Update traction of vertices.
  fa2UpdateTract(g, forceX, forceY, oldForceX, oldForceY, tra);

  // Update speed of vertices.
  fa2UpdateSpeed(g, speed, swg, forceX, forceY, graphSpeed);

  // Update displacement of vertices.
  fa2UpdateDisplacement(g, speed, forceX, forceY, dispX, dispY);

  // Set current forces as old forces in vertex data.
  fa2SaveOldForces(g, forceX, forceY, oldForceX, oldForceY);

  // Update vertex locations based on speed.
  fa2UpdateLocation(g, dispX, dispY);
}

void fa2RunOnGraph(Graph* g, unsigned int n)
{
  for (size_t i = 0; i < n; i++)
  {
    fa2RunOnce(g);
  }
}

