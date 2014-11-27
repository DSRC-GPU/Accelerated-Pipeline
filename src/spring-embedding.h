/*!
 \file spring-embedding.h
 Force Atlas 2 spring embedding implementation.
 */
#ifndef FORCE_ATLAS_2_H
#define FORCE_ATLAS_2_H

#include "graph.h"

#ifdef YIFAN_HU

#define YIFAN_HU_C 1.0
#define YIFAN_HU_K 1.0
#define YIFAN_HU_STEP 1.0
#define YIFAN_HU_T 0.9

#else // Force Atlas 2 constants

/*!
 * Repulsion constant.
 */
#define K_R 1.0

/*!
 * Force atlas 2 paramter.
 */
#define K_S 1.0

/*!
 * Gravity constant.
 */
#define K_G 1.0

/*!
 * Force atlas 2 parameter.
 */
#define K_SMAX 10.0

/*!
 * Force atlas 2 parameter.
 */
#define TAU 0.01

#endif

#ifdef YIFAN_HU
#define GLOBAL_VARS 4
#else // Force Atlas 2
#define GLOBAL_VARS 4
#endif

/*!
 * Small number that is assigned to variables when they currently equal 0,
 * but need to be used as a denumerator or otherwise cannot be 0.
 */
#define EPSILON 0.1

/*!
 * If two floats have a difference smaller than this number, we consider them to
 * be equal.
 */
#define FLOAT_EPSILON 0.0000001

/*!
 * All data requiered to perform Force Atlas 2 computations.
 * Used when preparing and cleaning memory.
 * All pointers in this struct are device pointers.
 */
typedef struct ForceAtlas2Data
{
  /*!
   * Array that holds the traction values for each vertex.
   */
  float* tra;
  /*!
   * Array that holds the swing values for each vertex.
   */
  float* swg;
  float* speed;
  /*!
   * Array that holds the x-force value for each vertex.
   */
  float* forceX;
  /*!
   * Array that holds the y-force value for each vertex.
   */
  float* forceY;
  /*!
   * Array that holds the x-force value for each vertex, from the previous iteration.
   */
  float* oldForceX;
  /*!
   * Array that holds the y-force value for each vertex, from the previous iteration.
   */
  float* oldForceY;
  /*!
   * Pointer to the graph speed value in the global memory on the device.
   */
  float* graphSpeed;
} ForceAtlas2Data;

/*!
 * Allocates memory for force atlas 2 computation.
 */
void fa2PrepareMemory(ForceAtlas2Data* data, unsigned int numvertices);

/*!
 * Cleans the memory reserved for the Force Atlas 2 computation.
 */
void fa2CleanMemory(ForceAtlas2Data* data);

/*!
 * Runs the Force Atlas 2 spring embedding on a graph.
 * \param[in,out] g The graph on which to run the spring embedding.
 * \param[in] n The number of iterations the algorithm should run.
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
