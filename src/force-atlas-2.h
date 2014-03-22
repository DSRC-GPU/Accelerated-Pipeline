
#ifndef FORCE_ATLAS_2_H
#define FORCE_ATLAS_2_H

#include "graph.h"
#include "forces.h"

#define FA2_NUMFORCES 3
#define K_R 1

extern simpleForce FA2_FORCES[];

// Gravity force
void fa2Gravity(Graph* g, Vertex* v);

// Repulsion between vertices
void fa2Repulsion(Graph* g, Vertex* v);

// Attraction on edges
void fa2Attraction(Graph* g, Vertex* v);

#endif

