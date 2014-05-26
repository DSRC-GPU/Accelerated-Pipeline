/*!
 * \file force-atlas-2.c
 * Sequential implementation for the Force Atlas 2 spring embedding algorithm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "force-atlas-2.h"
#include "math.h"
#include "timer.h"
#include "vector.h"

#define PRINTID 0

/*!
 * Calculates the number of neighbours for every node.
 * \param[in] g The input graph.
 * \param[out] numNeighbours An array containing the out-degree for each node in
 * the given graph.
 */
void calcNumNeighbours(Graph* g, unsigned int* numNeighbours);

/*!
 * Updates the current force on each vertex with the current gravity.
 * \param g The graph on which to apply the update.
 * \param forceX Array holding the x-direction forces on each vertex.
 * \param forceY Array holding the y-direction forces on each vertex.
 * \param deg Array holding the out-degree for each vertex.
 */
void fa2Gravity(Graph* g, float* forceX, float* forceY, unsigned int* deg);

/*!
 * Applies repulsion forces on each vertex.
 * \param g The graph that contains the vertices on which to apply the repulsion
 * forces.
 * \param forceX Array holding the x-direction forces on each vertex.
 * \param forceY Array holding the y-direction forces on each vertex.
 * \param deg Array holding the out-degree of each vertex.
 */
void fa2Repulsion(Graph* g, float* forceX, float* forceY, unsigned int* deg);

/*!
 * Applies attraction forces on all pairs of vertices that are connected by an
 * edge.
 * \param g The graph which holds the vertices and edges on which to apply the
 * attraction forces.
 * \param forceX Array holding the x-direction forces on each vertex.
 * \param forceY Array holding the y-direction forces on each vertex.
 */
void fa2Attraction(Graph* g, float* forceX, float* forceY);

/*!
 * Updates the swing value of each vertex in the Graph.
 * \param[in] g The graph to update.
 * \param[in] forceX Array holding the x-direction forces on each vertex.
 * \param[in] forceY Array holding the y-direction forces on each vertex.
 * \param[in] oldForceX Array holding the x-direction forces on each vertex for the
 * previous iteration.
 * \param[in] oldForceY Array holding the y-direction forces on each vertex for the
 * previous iteration.
 * \param[out] swg An array holding the calculated swing values for each vertex.
 */
void fa2UpdateSwing(Graph* g, float* forceX, float* forceY, float* oldForceX,
    float* oldForceY, float* swg);

/*!
 * Updates the traction value of each vertex in the Graph.
 * \param[in] g The graph to update.
 * \param[in] forceX Array holding the x-direction forces on each vertex.
 * \param[in] forceY Array holding the y-direction forces on each vertex.
 * \param[in] oldForceX Array holding the x-direction forces on each vertex for the
 * previous iteration.
 * \param[in] oldForceY Array holding the y-direction forces on each vertex for the
 * previous iteration.
 * \param[out] tra An array holding the calculated traction values for each
 * vertex.
 */
void fa2UpdateTract(Graph* g, float* forceX, float* forceY, float* oldForceX,
    float* oldForceY, float* tra);

void fa2UpdateSwingGraph(Graph*, float*, unsigned int*, float*);
void fa2UpdateTractGraph(Graph*, float*, unsigned int*, float*);
void fa2UpdateSpeedGraph(float, float, float*);
void fa2UpdateSpeed(Graph*, float*, float*, float*, float*, float);
void fa2UpdateDisplacement(Graph*, float*, float*, float*, float*, float*);
void fa2SaveOldForces(Graph*, float*, float*, float*, float*);
void fa2UpdateLocation(Graph*, float*, float*);

void calcNumNeighbours(Graph* g, unsigned int* deg)
{
  for (size_t i = 0; i < g->numedges; i++)
  {
    unsigned int node = g->edgeSources[i];
    deg[node]++; 
  }
}

void fa2Gravity(Graph* g, float* forceX, float* forceY, unsigned int* deg)
{
  if (!g) return;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    float vx = g->vertexXLocs[i];
    float vy = g->vertexYLocs[i];
    float vlen = vectorGetLength(vx, vy);
    vectorInverse(&vx, &vy);
    vectorMultiply(&vx, &vy, K_G * (deg[i] + 1) / vlen);
    vectorAdd(&forceX[i], &forceY[i], vx, vy);
    if (i == PRINTID)
      printf("g\t%f\t%f\n", vx, vy);
  }
}

void fa2Repulsion(Graph* g, float* forceX, float* forceY, unsigned int* deg)
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

      if (dist > 0)
      {
        vectorNormalize(&vx1, &vy1);
        vectorMultiply(&vx1, &vy1, K_R * (((deg[i] + 1) * (deg[j] + 1))
              / dist));
        // vectorMultiply(&vx1, &vy1, 0.5);

        vectorAdd(&forceX[i], &forceY[i], vx1, vy1);
        if (i == PRINTID)
          printf("r\t%i\t%f\t%f\n", j, vx1, vy1);
      }
      else if (i == PRINTID)
        printf("r\t%i\n", j);
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
    // vectorMultiply(&vx2, &vy2, 0.5);
    vectorAdd(&forceX[v1Index], &forceY[v1Index], vx2, vy2);
    if (v1Index == PRINTID)
      printf("a\t%f\t%f\n", vx2, vy2);
  }
}

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
void fa2UpdateSwing(Graph* g, float* forceX, float* forceY,
    float* oldForceX, float* oldForceY, float* swg)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    float fx = oldForceX[i];
    float fy = oldForceY[i];
    vectorSubtract(&fx, &fy, forceX[i], forceY[i]);
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
    float fx = oldForceX[i];
    float fy = oldForceY[i];
    vectorAdd(&fx, &fy, forceX[i], forceY[i]);
    float vlen = vectorGetLength(fx, fy);
    tra[i] = vlen / 2;
  }
}

// Calculate the current swing of the graph.
void fa2UpdateSwingGraph(Graph* g, float* swg, unsigned int* deg, float* gswing)
{
  *gswing = 0;
  for (size_t i = 0; i < g->numvertices; i++)
  {
    *gswing += (deg[i] + 1) * swg[i];
  }
}

// Calculate the current traction of the graph.
void fa2UpdateTractGraph(Graph* g, float* tra, unsigned int* deg, float* gtract)
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

  if (gswing == 0)
    gswing = FLOAT_EPSILON;

  *gspeed = TAU * (gtract / gswing);

  //if (*gspeed <= 0)
  //  *gspeed = EPSILON;
  // Do not allow more then 50% speed increase.
  if (oldSpeed > 0 && *gspeed > 1.5 * oldSpeed)
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
    //speed[i] = fmin(speed[i],
    //    K_SMAX / vForceLen);
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
  static unsigned int* numNeighbours = NULL;
  static float* tra = NULL;
  static float* swg = NULL;
  static float* speed = NULL;
  static float* forceX = NULL;
  static float* forceY = NULL;
  static float* oldForceX = NULL;
  static float* oldForceY = NULL;
  static float* dispX = NULL;
  static float* dispY = NULL;

  static Timer timer;

  float graphSwing = 0.0;
  float graphTract = 0.0;
  static float graphSpeed = 0.0;

  if (firstRun)
  {
    numNeighbours = (unsigned int*) calloc(g->numvertices, sizeof(unsigned int));
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

  // Reset forces on vertices to 0.
  memset(forceX, 0, sizeof(float) * g->numvertices);
  memset(forceY, 0, sizeof(float) * g->numvertices);

  // Gravity force
  startTimer(&timer);
  fa2Gravity(g, forceX, forceY, numNeighbours);
  stopTimer(&timer);
  //printf("time: gravity.\n");
  //printTimer(&timer);
  // Repulsion between vertices
  startTimer(&timer);
  fa2Repulsion(g, forceX, forceY, numNeighbours);
  stopTimer(&timer);
  //printf("time: repulsion.\n");
  //printTimer(&timer);
  // Attraction on edges
  startTimer(&timer);
  fa2Attraction(g, forceX, forceY);
  stopTimer(&timer);
  //printf("time: attraction.\n");
  //printTimer(&timer);

  // Calculate speed of vertices.
  // Update swing of vertices.
  fa2UpdateSwing(g, forceX, forceY, oldForceX, oldForceY, swg);

  // Update traction of vertices.
  fa2UpdateTract(g, forceX, forceY, oldForceX, oldForceY, tra);

  // Update swing of Graph.
  fa2UpdateSwingGraph(g, swg, numNeighbours, &graphSwing);

  // Update traction of Graph.
  fa2UpdateTractGraph(g, tra, numNeighbours, &graphTract);

  // Update speed of Graph.
  fa2UpdateSpeedGraph(graphSwing, graphTract, &graphSpeed);

  // Update speed of vertices.
  fa2UpdateSpeed(g, speed, swg, forceX, forceY, graphSpeed);

  // Update displacement of vertices.
  fa2UpdateDisplacement(g, speed, forceX, forceY, dispX, dispY);

  // Update vertex locations based on speed.
  startTimer(&timer);
  fa2UpdateLocation(g, dispX, dispY);

  stopTimer(&timer);
  //printf("time: moving vertices.\n");
  //printTimer(&timer);

  // Set current forces as old forces in vertex data.
  fa2SaveOldForces(g, forceX, forceY, oldForceX, oldForceY);
}

void fa2RunOnGraph(Graph* g, unsigned int n)
{
  Timer timer;
  for (size_t i = 0; i < n; i++)
  {
    startTimer(&timer);
    fa2RunOnce(g);
    stopTimer(&timer);
    printGraph(g);
    printf("\n");
    //printf("time: iteration.\n");
    //printTimer(&timer);
  }
}

