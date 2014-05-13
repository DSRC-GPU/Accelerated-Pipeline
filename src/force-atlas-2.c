
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "force-atlas-2.h"
#include "math.h"
#include "vector.h"

#define K_R 1.0
#define K_S 1.0
#define K_G 1.0
#define K_SMAX 10.0
#define TAU 0.01
#define EPSILON 0.1
#define FLOAT_EPSILON 0.0000001

// Calculate the number of neighbours for every node.
void calcNumNeighbours(Graph*, unsigned int*);
// Gravity force
void fa2Gravity(Graph*, double*, double*, unsigned int*);
// Repulsion between vertices
void fa2Repulsion(Graph*, double*, double*, unsigned int*);
// Attraction on edges
void fa2Attraction(Graph*, double*, double*);

void fa2UpdateSwing(Graph*, double*, double*, double*, double*, double*);
void fa2UpdateTract(Graph*, double*, double*, double*, double*, double*);
void fa2UpdateSwingGraph(Graph*, double*, unsigned int*, double*);
void fa2UpdateTractGraph(Graph*, double*, unsigned int*, double*);
void fa2UpdateSpeedGraph(double, double, double*);
void fa2UpdateSpeed(Graph*, double*, double*, double*, double*, double);
void fa2UpdateDisplacement(Graph*, double*, double*, double*, double*, double*);
void fa2SaveOldForces(Graph*, double*, double*, double*, double*);
void fa2UpdateLocation(Graph*, double*, double*);

void calcNumNeighbours(Graph* g, unsigned int* deg)
{
  for (size_t i = 0; i < g->numedges; i++)
  {
    unsigned int node = g->edgeSources[i];
    deg[node]++; 
  }
}

void fa2Gravity(Graph* g, double* forceX, double* forceY, unsigned int* deg)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    double vx = g->vertexXLocs[i];
    double vy = g->vertexYLocs[i];
    double vlen = vectorGetLength(vx, vy);
    printf("grav force: %f,%f.\n", vx, vy);
    printf("vlen: %f.\n", vlen);
    vectorInverse(&vx, &vy);
    vectorMultiply(&vx, &vy, K_G * (deg[i] + 1) / vlen);
    vectorAdd(&forceX[i], &forceY[i], vx, vy);
    printf("grav force: %f,%f.\n", vx, vy);
  }
}

void fa2Repulsion(Graph* g, double* forceX, double* forceY, unsigned int* deg)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    for (size_t j = 0; j < g->numvertices; j++)
    {
      if (i == j) continue;
      double vx1 = g->vertexXLocs[i];
      double vy1 = g->vertexYLocs[i];
      double vx2 = g->vertexXLocs[j];
      double vy2 = g->vertexYLocs[j];

      vectorSubtract(&vx1, &vy1, vx2, vy2);
      double dist = vectorGetLength(vx1, vy1);

      if (dist > 0)
      {
        vectorNormalize(&vx1, &vy1);
        vectorMultiply(&vx1, &vy1, K_R * (((deg[i] + 1) * (deg[j] + 1))
                / dist));
        // vectorMultiply(&vx1, &vy1, 0.5);

        vectorAdd(&forceX[i], &forceY[i], vx1, vy1);
      }
    }
  }
}

void fa2Attraction(Graph* g, double* forceX, double* forceY)
{
  if (!g) return;
  for (size_t i = 0; i < g->numedges; i++)
  {
    int v1Index = g->edgeSources[i];
    int v2Index = g->edgeTargets[i];

    double vx1 = g->vertexXLocs[v1Index];
    double vy1 = g->vertexYLocs[v1Index];
    double vx2 = g->vertexXLocs[v2Index];
    double vy2 = g->vertexYLocs[v2Index];

    vectorSubtract(&vx2, &vy2, vx1, vy1);
    // vectorMultiply(&vx2, &vy2, 0.5);
    vectorAdd(&forceX[v1Index], &forceY[v1Index], vx2, vy2);
  }
}

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
void fa2UpdateSwing(Graph* g, double* forceX, double* forceY,
    double* oldForceX, double* oldForceY, double* swg)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    double fx = oldForceX[i];
    double fy = oldForceY[i];
    printf("old forces x:y => %f:%f.\n", fx, fy);
    vectorSubtract(&fx, &fy, forceX[i], forceY[i]);
    double vlen = vectorGetLength(fx, fy);
    swg[i] = vlen;
  }
}

// Updates the traction for each vertex, as described in the Force Atlas 2
// paper.
void fa2UpdateTract(Graph* g, double* forceX, double* forceY,
    double* oldForceX, double* oldForceY, double* tra)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    double fx = forceX[i];
    double fy = forceY[i];
    vectorAdd(&fx, &fy, oldForceX[i], oldForceY[i]);
    double vlen = vectorGetLength(fx, fy);
    tra[i] = vlen / 2;
  }
}

// Calculate the current swing of the graph.
void fa2UpdateSwingGraph(Graph* g, double* swg, unsigned int* deg, double* gswing)
{
  *gswing = 0;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    *gswing += (deg[i] + 1) * swg[i];
  }
  printf("Swing: %f.\n", *gswing);
}

// Calculate the current traction of the graph.
void fa2UpdateTractGraph(Graph* g, double* tra, unsigned int* deg, double* gtract)
{
  *gtract = 0;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    *gtract += (deg[i] + 1) * tra[i];
  }
}

void fa2UpdateSpeedGraph(double gswing, double gtract, double* gspeed)
{
  double oldSpeed = *gspeed;

  if (gswing == 0)
    gswing = FLOAT_EPSILON;

  *gspeed = TAU * (gtract / gswing);

  //if (*gspeed <= 0)
  //  *gspeed = EPSILON;
  // Do not allow more then 50% speed increase.
  if (oldSpeed > 0 && *gspeed > 1.5 * oldSpeed)
    *gspeed = 1.5 * oldSpeed;
  printf("GSpeed: %f.\n", *gspeed);
}

void fa2UpdateSpeed(Graph* g, double* speed, double* swg, double* forceX,
    double* forceY, double gs)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    double vSwg = swg[i];
    if (vSwg <= 0)
      vSwg = EPSILON;
    double vForceLen = vectorGetLength(forceX[i], forceY[i]);
    if (vForceLen <= 0)
      vForceLen = EPSILON;

    speed[i] = K_S * gs / (1 + (gs * sqrt(vSwg)));
    //speed[i] = fmin(speed[i],
    //    K_SMAX / vForceLen);
  }
}

// Save current forces as the previous forces for the next tick.
void fa2SaveOldForces(Graph* g, double* forceX, double* forceY,
    double* oldForceX, double* oldForceY)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    oldForceX[i] = forceX[i];
    oldForceY[i] = forceY[i];
  }
}

void fa2UpdateDisplacement(Graph* g, double* speed, double* forceX,
    double* forceY, double* dispX, double* dispY)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    dispX[i] = forceX[i];
    dispY[i] = forceY[i];
    vectorMultiply(&dispX[i], &dispY[i], speed[i]);
  }
}

void fa2UpdateLocation(Graph* g, double* xdisp, double* ydisp)
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
  static unsigned int* numNeighbours = NULL;
  static double* tra = NULL;
  static double* swg = NULL;
  static double* speed = NULL;
  static double* forceX = NULL;
  static double* forceY = NULL;
  static double* oldForceX = NULL;
  static double* oldForceY = NULL;
  static double* dispX = NULL;
  static double* dispY = NULL;

  double graphSwing = 0.0;
  double graphTract = 0.0;
  static double graphSpeed = 0.0;

  if (firstRun)
  {
    numNeighbours = (unsigned int*) calloc(g->numvertices, sizeof(unsigned int));
    calcNumNeighbours(g, numNeighbours);
    tra = (double*) calloc(g->numvertices, sizeof(double));
    swg = (double*) calloc(g->numvertices, sizeof(double));
    speed = (double*) calloc(g->numvertices, sizeof(double));
    forceX = (double*) calloc(g->numvertices, sizeof(double));
    forceY = (double*) calloc(g->numvertices, sizeof(double));
    oldForceX = (double*) calloc(g->numvertices, sizeof(double));
    oldForceY = (double*) calloc(g->numvertices, sizeof(double));
    dispX = (double*) calloc(g->numvertices, sizeof(double));
    dispY = (double*) calloc(g->numvertices, sizeof(double));

    firstRun = 0;
  }

  // Update swing of Graph.
  fa2UpdateSwingGraph(g, swg, numNeighbours, &graphSwing);

  printf("swing: %f.\n", graphSwing);

  // Update traction of Graph.
  fa2UpdateTractGraph(g, tra, numNeighbours, &graphTract);

  // Update speed of Graph.
  fa2UpdateSpeedGraph(graphSwing, graphTract, &graphSpeed);

  // Reset forces on vertices to 0.
  memset(forceX, 0, sizeof(double) * g->numvertices);
  memset(forceY, 0, sizeof(double) * g->numvertices);

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
    printGraph(g);
    printf("\n");
  }
}

