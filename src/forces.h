
#ifndef FORCES_H
#define FORCES_H

typedef void (*simpleForce)(Graph*, Vertex*);
typedef void (*dataForce)(Graph*, Vertex*, void*);

typedef struct ForceWithData
{
  dataForce force;
  void* data;
} ForceWithData;

// Give a set of forces that are to be used by the spring embedding core.
void runForcesOnGraph(Graph*, unsigned int, simpleForce*);
void runDataForcesOnGraph(Graph*, unsigned int, ForceWithData*);

#endif

