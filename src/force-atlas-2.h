/*!
  \file force-atlas-2.h
  Force Atlas 2 spring embedding implementation.
 */
#ifndef FORCE_ATLAS_2_H
#define FORCE_ATLAS_2_H

#include "graph.h"

/*!
  Runs the Force Atlas 2 spring embedding on a graph.
  \param g The graph on which to run the spring embedding.
  \param n The number of iterations the algorithm should run.
 */
void fa2RunOnGraph(Graph* g, unsigned int n);

#endif

