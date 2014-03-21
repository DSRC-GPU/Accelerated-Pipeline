
void applyForceOnGraph(Graph*, simpleForce)
void applyDataForceOnGraph(Graph*, ForceWithData*)

// Give a set of forces that are to be used by the spring embedding core.
void runForcesOnGraph(Graph*, unsigned int, simpleForce*)
void runDataForcesOnGraph(Graph*, unsigned int, ForceWithData*)

void applyForceOnGraph(Graph* g, simpleForce sf)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    sf(g, &g->vertices[i]);
  }
}

void applyDataForceOnGraph(Graph* g, ForceWithData* fwd)
{
  for (size_t i = 0; i < g->numvertices; i++)
  {
    fwd->force(g, &g->vertices[i], fwd->data);
  }
}

// Give a set of forces that are to be used by the spring embedding core.
void runForcesOnGraph(Graph* g, unsigned int num, simpleForce* sf)
{
  for (size_t i = 0; i < num; i++)
  {
    applyForceOnGraph(g, sf[i]);
  }
}

void runDataForcesOnGraph(Graph* g, unsigned int num, ForceWithData* fwd)
{
  for (size_t i = 0; i < num; i++)
  {
    applyDataForceOnGraph(g, fwd[i]);
  }
}

