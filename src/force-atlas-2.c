/*!
 * \file force-atlas-2.c
 * Sequential implementation for the Force Atlas 2 spring embedding algorithm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include "force-atlas-2.h"
#include "math.h"
#include "timer.h"
#include "vector.h"
#include "util.h"

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
 *    forces.
 * \param forceX Array holding the x-direction forces on each vertex.
 * \param forceY Array holding the y-direction forces on each vertex.
 * \param deg Array holding the out-degree of each vertex.
 */
void fa2Repulsion(Graph* g, float* forceX, float* forceY, unsigned int* deg);

/*!
 * Applies attraction forces on all pairs of vertices that are connected by an
 * edge.
 * \param g The graph which holds the vertices and edges on which to apply the
 *    attraction forces.
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
 *    previous iteration.
 * \param[in] oldForceY Array holding the y-direction forces on each vertex for the
 *    previous iteration.
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
 *    previous iteration.
 * \param[in] oldForceY Array holding the y-direction forces on each vertex for the
 *    previous iteration.
 * \param[out] tra An array holding the calculated traction values for each
 *    vertex.
 */
void fa2UpdateTract(Graph* g, float* forceX, float* forceY, float* oldForceX,
    float* oldForceY, float* tra);

/*!
 * Update the swing value of the Graph itself.
 * \param[in] g The graph whose swing value should be calculated.
 * \param[in] swg The array holding the swing values for each vertex in the
 *    graph.
 * \param[in] deg The array holding the out degree values for each vertex in the
 *    graph.
 * \param[out] gSwing A float pointer where the graph swing value should be
 *    stored.
 */
void fa2UpdateSwingGraph(Graph* g, float* swg, unsigned int* deg, float* gSwing);

/*!
 * Update the traction value of the graph itself.
 * \param[in] g The graph whoses traction should be calculated.
 * \param[in] tra The array holding the traction values for each vertex in the
 *    graph.
 * \param[in] deg The array holding the out degree values for each vertex in the
 *    graph.
 * \param[out] gTract A float pointer where the graph traction value should be
 *    stored.
 */
void fa2UpdateTractGraph(Graph* g, float* tra, unsigned int* deg, float* gTract);

/*!
 * Updates the speed value of the graph itself.
 * \param[in] gSwing The graph swing value.
 * \param[in] gTract The graph traction value.
 * \param[out] gSpeed The graph speed value.
 */
void fa2UpdateSpeedGraph(float gSwing, float gTract, float* gSpeed);

/*!
 * Updates the speed value for each vertex in the graph.
 * \param[in] g The graph.
 * \param[out] speed The array where the speed values should be stored.
 * \param[in] swg The swing values for all vertices in the graph.
 * \param[in] forceX The x forces for all vertices in the graph.
 * \param[in] forceY The y forces for all vertices in the graph.
 * \param[in] gs The graph speed value.
 */
void fa2UpdateSpeed(Graph* g, float* speed, float* swg, float* forceX,
    float* forceY, float gs);

/*!
 * Updates the displacement value for each vertex in the graph.
 * \param[in] g The graph.
 * \param[in] speed The array that holds the speed values for each vertex.
 * \param[in] forceX The x forces for each vertex in the graph.
 * \param[in] forceY The y forces for each vertex in the graph.
 * \param[out] dispX The array where the x displacement for each vertex should
 *    be stored.
 * \param[out] dispY The array where the y displacement for each vertex should
 *    be stored.
 */
void fa2UpdateDisplacement(Graph*, float*, float*, float*, float*, float*);

/*!
 * Stores the forces from the current iteration as old forces. Deletes the
 * forces from the previous iteration.
 * 
 * \param[in] g The graph.
 * \param[in] forceX The x forces of the current iteration.
 * \param[in] forceY The y forces of the current iteration.
 * \param[out] oldForceX The array that should be used to store the x forces of
 *    the current iteration.
 * \param[out] oldForceY The array that should be used to store the y forces of
 *    the current iteration.
 */
void fa2SaveOldForces(Graph*, float*, float*, float*, float*);

/*!
 * Updates the location of each vertex according to the given displacement.
 *
 * \param[in] g The graph.
 * \param[in] xdisp The array of x displacement values for each vertex.
 * \param[in] ydisp The array of y displacement values for each vertex.
 */
void fa2UpdateLocation(Graph* g, float* xdisp, float* ydisp);

void fa2Gravity(Graph* g, float* forceX, float* forceY, unsigned int* deg)
{
  if (!g)
    return;
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    float vx = g->vertices->vertexXLocs[i];
    float vy = g->vertices->vertexYLocs[i];
    float vlen = vectorGetLength(vx, vy);
    assert(vlen != 0);
    vectorInverse(&vx, &vy);
    vectorMultiply(&vx, &vy, K_G * (deg[i] + 1) / vlen);
    if (i == 0)
    {
      DEBUG_PRINT("g:%f\n", vx);
    }
    vectorAdd(&forceX[i], &forceY[i], vx, vy);
  }
}

void fa2Repulsion(Graph* g, float* forceX, float* forceY, unsigned int* deg)
{
  if (!g)
    return;
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    for (size_t j = 0; j < g->vertices->numvertices; j++)
    {
      if (i == j)
        continue;
      float vx1 = g->vertices->vertexXLocs[i];
      float vy1 = g->vertices->vertexYLocs[i];
      float vx2 = g->vertices->vertexXLocs[j];
      float vy2 = g->vertices->vertexYLocs[j];

      vectorSubtract(&vx1, &vy1, vx2, vy2);
      float dist = vectorGetLength(vx1, vy1);

      if (dist > 0)
      {
        vectorNormalize(&vx1, &vy1);
        vectorMultiply(&vx1, &vy1,
            K_R * (((deg[i] + 1) * (deg[j] + 1)) / dist));
        // vectorMultiply(&vx1, &vy1, 0.5);

        vectorAdd(&forceX[i], &forceY[i], vx1, vy1);
      }
    }
  }
}

void fa2Attraction(Graph* g, float* forceX, float* forceY)
{
  if (!g)
    return;

  float* vxLocs = g->vertices->vertexXLocs;
  float* vyLocs = g->vertices->vertexYLocs;

  for (size_t gid = 0; gid < g->vertices->numvertices; gid++)
  {
    float vx1 = vxLocs[gid];
    float vy1 = vyLocs[gid];
    // Each thread goes through its array of edges.
    unsigned int maxedges = g->edges->numedges[gid];
    for (size_t i = 0; i < maxedges; i++)
    {
      unsigned int index = gid + (g->vertices->numvertices * i);
      unsigned int target = g->edges->edgeTargets[index];
      if (target == UINT_MAX)
        continue;
      // Compute attraction force.
      float vx2 = vxLocs[target];
      float vy2 = vyLocs[target];

      vectorSubtract(&vx2, &vy2, vx1, vy1);
      // vectorMultiply(&vx2, &vy2, 0.5);
      vectorAdd(&forceX[gid], &forceY[gid], vx2, vy2);
      if (gid == 0)
      {
        DEBUG_PRINT("a:%f\t%u\n", vx2, target);
      }
    }
  }
}

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
void fa2UpdateSwing(Graph* g, float* forceX, float* forceY, float* oldForceX,
    float* oldForceY, float* swg)
{
  for (size_t i = 0; i < g->vertices->numvertices; i++)
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
void fa2UpdateTract(Graph* g, float* forceX, float* forceY, float* oldForceX,
    float* oldForceY, float* tra)
{
  for (size_t i = 0; i < g->vertices->numvertices; i++)
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
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    *gswing += (deg[i] + 1) * swg[i];
  }
}

// Calculate the current traction of the graph.
void fa2UpdateTractGraph(Graph* g, float* tra, unsigned int* deg, float* gtract)
{
  *gtract = 0;
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    *gtract += (deg[i] + 1) * tra[i];
  }
}

void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed)
{
  float oldSpeed = *gspeed;

  if (gswing == 0)
  {
    gswing = FLOAT_EPSILON;
  }

  *gspeed = TAU * (gtract / gswing);

  if (oldSpeed > 0 && *gspeed > 1.5 * oldSpeed)
  {
    *gspeed = 1.5 * oldSpeed;
  }
}

void fa2UpdateSpeed(Graph* g, float* speed, float* swg, float* forceX,
    float* forceY, float gs)
{
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    float vSwg = swg[i];
    if (vSwg <= 0)
      vSwg = EPSILON;
    float vForceLen = vectorGetLength(forceX[i], forceY[i]);
    if (vForceLen <= 0)
      vForceLen = EPSILON;

    speed[i] = K_S * gs / (1 + (gs * sqrt(vSwg)));
  }
}

// Save current forces as the previous forces for the next tick.
void fa2SaveOldForces(Graph* g, float* forceX, float* forceY, float* oldForceX,
    float* oldForceY)
{
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    oldForceX[i] = forceX[i];
    oldForceY[i] = forceY[i];
  }
}

void fa2UpdateDisplacement(Graph* g, float* speed, float* forceX, float* forceY,
    float* dispX, float* dispY)
{
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    dispX[i] = forceX[i];
    dispY[i] = forceY[i];
    vectorMultiply(&dispX[i], &dispY[i], speed[i]);
  }
}

void fa2UpdateLocation(Graph* g, float* xdisp, float* ydisp)
{
  for (size_t i = 0; i < g->vertices->numvertices; i++)
  {
    g->vertices->vertexXLocs[i] += xdisp[i];
    g->vertices->vertexYLocs[i] += ydisp[i];
  }
}

/*!
 * Runs one iteration of the force atlas 2 spring embedding on the given graph.
 *
 * \param[in,out] g The graph to use for force atlas 2.
 */
void fa2RunOnce(Graph* g)
{
  static int firstRun = 1;
  static float* tra = NULL;
  static float* swg = NULL;
  static float* speed = NULL;
  static float* forceX = NULL;
  static float* forceY = NULL;
  static float* oldForceX = NULL;
  static float* oldForceY = NULL;
  static float* dispX = NULL;
  static float* dispY = NULL;

  Timer* timer = timerNew();

  float graphSwing = 0.0;
  float graphTract = 0.0;
  static float graphSpeed = 0.0;

  if (firstRun)
  {
    tra = (float*) calloc(g->vertices->numvertices, sizeof(float));
    swg = (float*) calloc(g->vertices->numvertices, sizeof(float));
    speed = (float*) calloc(g->vertices->numvertices, sizeof(float));
    forceX = (float*) calloc(g->vertices->numvertices, sizeof(float));
    forceY = (float*) calloc(g->vertices->numvertices, sizeof(float));
    oldForceX = (float*) calloc(g->vertices->numvertices, sizeof(float));
    oldForceY = (float*) calloc(g->vertices->numvertices, sizeof(float));
    dispX = (float*) calloc(g->vertices->numvertices, sizeof(float));
    dispY = (float*) calloc(g->vertices->numvertices, sizeof(float));

    firstRun = 0;
  }

  // Reset forces on vertices to 0.
  memset(forceX, 0, sizeof(float) * g->vertices->numvertices);
  memset(forceY, 0, sizeof(float) * g->vertices->numvertices);

  // Gravity force
  startTimer(timer);
  fa2Gravity(g, forceX, forceY, g->edges->numedges);
  stopTimer(timer);
  DEBUG_PRINT_DEVICE(forceX, g->vertices->numvertices);

  printTimer(timer, "force: gravity");
  // Repulsion between vertices
  startTimer(timer);
  fa2Repulsion(g, forceX, forceY, g->edges->numedges);
  stopTimer(timer);
  DEBUG_PRINT_DEVICE(forceX, g->vertices->numvertices);

  printTimer(timer, "force: repulsion");
  // Attraction on edges
  startTimer(timer);
  fa2Attraction(g, forceX, forceY);
  stopTimer(timer);
  DEBUG_PRINT_DEVICE(forceX, g->vertices->numvertices);

  printTimer(timer, "force: attraction");

  // Calculate speed of vertices.
  // Update swing of vertices.
  fa2UpdateSwing(g, forceX, forceY, oldForceX, oldForceY, swg);

  // Update traction of vertices.
  fa2UpdateTract(g, forceX, forceY, oldForceX, oldForceY, tra);

  DEBUG_PRINT_DEVICE(forceX, g->vertices->numvertices);

  // Update swing of Graph.
  fa2UpdateSwingGraph(g, swg, g->edges->numedges, &graphSwing);

  // Update traction of Graph.
  fa2UpdateTractGraph(g, tra, g->edges->numedges, &graphTract);

  // Update speed of Graph.
  fa2UpdateSpeedGraph(graphSwing, graphTract, &graphSpeed);

  // Update speed of vertices.
  fa2UpdateSpeed(g, speed, swg, forceX, forceY, graphSpeed);

  // Update displacement of vertices.
  fa2UpdateDisplacement(g, speed, forceX, forceY, dispX, dispY);

  // Update vertex locations based on speed.
  startTimer(timer);
  fa2UpdateLocation(g, dispX, dispY);

  stopTimer(timer);
  printTimer(timer, "moving vertices");

  // Set current forces as old forces in vertex data.
  fa2SaveOldForces(g, forceX, forceY, oldForceX, oldForceY);

  timerClean(timer);
}

void fa2RunOnGraph(Graph* g, unsigned int n)
{
  Timer* timer = timerNew();
  for (size_t i = 0; i < n; i++)
  {
    startTimer(timer);
    fa2RunOnce(g);
    stopTimer(timer);
    printTimer(timer, "embedding iteration");
  }
  timerClean(timer);
}

