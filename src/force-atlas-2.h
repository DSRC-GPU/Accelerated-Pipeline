
// Gravity force
void fa2Gravity(Graph* g, Vertex* v);

// Repulsion between vertices
void fa2Repulsion(Graph* g, Vertex* v);

// Attraction on edges
void fa2Attraction(Graph* g, Vertex* v);

unsigned int FA2_NUMFORCES = 3;
simpleForce FA2_FORCES[3] = { fa2Gravity, fa2Repulsion, fa2Attraction };
