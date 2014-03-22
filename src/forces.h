
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

// Update the speed of vertices based on the their computed forces.
void updateSpeedOnGraph(Graph*);

// Update the location of vertices based on their computed speeds.
void updateLocationOnGraph(Graph*);

// Reset the computed forces on vertices to zero for the next iteration.
void resetForcesOnGraph(Graph*);

void getVectorBetweenVertex(Vertex*, Vertex*, Vector*);
float getVectorLength(Vector* v);
void addVectors(Vector*, Vector*);
void subtractVectors(Vector*, Vector*);
void normalizeVector(Vector*);
void inverseVector(Vector*);
void multiplyVector(Vector*, float);

#endif

