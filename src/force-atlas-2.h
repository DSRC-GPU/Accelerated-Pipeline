/*!
  \file force-atlas-2.h
  Force Atlas 2 spring embedding implementation.
 */
#ifndef FORCE_ATLAS_2_H
#define FORCE_ATLAS_2_H

#include "graph.h"

/*!
  Repulsion constant.
 */
#define K_R 1.0
#define K_S 1.0

/*!
  Gravity constant.
 */
#define K_G 1.0
#define K_SMAX 10.0
#define TAU 0.01
#define EPSILON 0.1

/*!
  If two floats have a difference smaller than this number, we consider them to
  be equal.
 */
#define FLOAT_EPSILON 0.0000001

/*!
  Runs the Force Atlas 2 spring embedding on a graph.
  \param g The graph on which to run the spring embedding.
  \param n The number of iterations the algorithm should run.
 */
void fa2RunOnGraph(Graph* g, unsigned int n);

#endif

