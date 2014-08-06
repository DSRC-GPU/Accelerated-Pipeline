/*!
 \file force-atlas-2.h
 Force Atlas 2 spring embedding implementation.
 */
#ifndef FORCE_ATLAS_2_H
#define FORCE_ATLAS_2_H

#include "graph.h"

/*!
 * Repulsion constant.
 */
#define K_R 1.0
#define K_S 1.0

/*!
 * Gravity constant.
 */
#define K_G 1.0
#define K_SMAX 10.0
#define TAU 0.01
#define EPSILON 0.1

/*!
 * If two floats have a difference smaller than this number, we consider them to
 * be equal.
 */
#define FLOAT_EPSILON 0.0000001

/*!
 * All data requiered to perform Force Atlas 2 computations.
 * Used when preparing and cleaning memory.
 */
typedef struct ForceAtlas2Data
{
  float* tra;
  float* swg;
  float* forceX;
  float* forceY;
  float* oldForceX;
  float* oldForceY;
  float* graphSwing;
  float* graphTract;
  float* graphSpeed;

  float* vxLocs;
  float* vyLocs;
  unsigned int* numEdges;
  unsigned int** edgeTargets;
} ForceAtlas2Data;

/*!
 * Allocates memory for force atlas 2 computation.
 */
void fa2PrepareMemory(ForceAtlas2Data* data, unsigned int numvertices,
    unsigned int numedges);

/*!
 * Cleans the memory reserved for the Force Atlas 2 computation.
 */
void fa2CleanMemory(ForceAtlas2Data* data);

/*!
 * Runs the Force Atlas 2 spring embedding on a graph.
 * \param g The graph on which to run the spring embedding.
 * \param n The number of iterations the algorithm should run.
 */
void fa2RunOnGraph(Graph* g, unsigned int n);

/*!
 * Runs the Force Atlas 2 spring embedding on all graphs in separate CUDA streams.
 *
 * \param[in] verticesIn An array of the vertices in the graph.
 * \param[in] edges An array of Edges pointers. This array is in fact the 'window'.
 * \param[in] numgraphs the number of graphs in the array.
 * \param[in] iterations the number of iterations the spring embedding should run.
 * \param[out] averageSpeedX the x-coordinates for the average speed vectors for all vertices.
 * \param[out] averageSpeedY the y-coordinates for the average speed vectors for all vertices.
 */
void fa2RunOnGraphInStream(Vertices* verticesIn, Edges** edges,
    unsigned int numgraphs, unsigned int iterations, float** averageSpeedX,
    float** averageSpeedY);

#endif
