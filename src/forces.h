
#ifndef FORCES_H
#define FORCES_H

#include "graph.h"

typedef void (*simpleForce)(Graph*, Vertex*);
typedef void (*dataForce)(Graph*, Vertex*, void*);

typedef struct ForceWithData
{
  dataForce force;
  void* data;
} ForceWithData;

// Give a set of forces that are to be used by the spring embedding core.
void updateForcesOnGraph(Graph*, unsigned int, simpleForce*);
void updateDataForcesOnGraph(Graph*, unsigned int, ForceWithData*);
void updateSpeedOnGraph(Graph*);
void updateLocationOnGraph(Graph*);
void resetForcesOnGraph(Graph*);

void getVectorBetweenVertex(Vertex*, Vertex*, Vector*);
void addVectors(Vector*, Vector*);
void subtractVectors(Vector*, Vector*);

#endif

