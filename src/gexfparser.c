#include "gexfparser.h"
#include "timer.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#define NODE_START_X 10
#define NODE_START_Y 10

xmlNode* xmlwGetChild(xmlNode* node, const char* name)
{
  xmlNode* child = node->children;
  while (child)
  {
    if (!strcmp((const char*) child->name, name))
    {
      return child;
    }
    child = child->next;
  }
  return NULL ;
}

unsigned int xmlwGetNumNodes(xmlNode* gexf)
{
  if (!gexf)
    return -1;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph)
    return -1;
  xmlNode* nodes = xmlwGetChild(graph, "nodes");
  if (!nodes)
    return -1;
  return atoi((const char*) xmlGetProp(nodes, (const xmlChar*) "count"));
}

unsigned int xmlwGetNumEdges(xmlNode* gexf)
{
  if (!gexf)
    return -1;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph)
    return -1;
  xmlNode* edges = xmlwGetChild(graph, "edges");
  if (!edges)
    return -1;
  return atoi((const char*) xmlGetProp(edges, (const xmlChar*) "count"));
}

unsigned int xmlwGetMaxEdges(xmlNode* gexf, unsigned int numvertices)
{
  if (!gexf)
    return -1;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph)
    return -1;
  xmlNode* edges = xmlwGetChild(graph, "edges");
  if (!edges)
    return -1;
  unsigned int* numedges = (unsigned int*) calloc(numvertices,
      sizeof(unsigned int));
  xmlNode* node = xmlwGetChild(edges, "edge");
  while (node)
  {
    unsigned int source = atoi(
        (const char*) xmlGetProp(node, (const xmlChar*) "source"));
    unsigned int target = atoi(
        (const char*) xmlGetProp(node, (const xmlChar*) "target"));
    numedges[source]++;
    numedges[target]++;
    node = node->next;
  }
  unsigned int max = 0;
  for (size_t i = 0; i < numvertices; i++)
  {
    if (numedges[i] > max)
      max = numedges[i];
  }
  free(numedges);
  return max;
}

void gexfParseVertex(xmlNode* n, float* vertexXLoc, float* vertexYLoc,
    int *vertexIds)
{
  if (!n || !vertexXLoc || !vertexYLoc)
    return;
  const char* id = (const char*) xmlGetProp(n, (const xmlChar*) "id");
  assert(id != NULL);
  *vertexIds = atoi(id);
  *vertexXLoc = NODE_START_X * (1 + *vertexIds);
  *vertexYLoc = NODE_START_Y * (1 + *vertexIds);
}

void gexfParseEdge(xmlNode* n, Graph* graph)
{
  if (!n || !graph)
    return;
  unsigned int source = atoi(
      (const char*) xmlGetProp(n, (const xmlChar*) "source"));
  unsigned int target = atoi(
      (const char*) xmlGetProp(n, (const xmlChar*) "target"));
  graphAddEdgeToVertex(graph, source, target);
  graphAddEdgeToVertex(graph, target, source);
}

void gexfParseVertices(xmlNode* gexf, float* vertexXLocs, float* vertexYLocs,
    int *vertexIds)
{
  if (!gexf || !vertexXLocs || !vertexYLocs)
    return;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph)
    return;
  xmlNode* nodes = xmlwGetChild(graph, "nodes");
  if (!nodes)
    return;

  size_t i = 0;
  xmlNode* node = nodes->children;
  while (node)
  {
    gexfParseVertex(node, &vertexXLocs[i], &vertexYLocs[i], &vertexIds[i]);
    i++;
    node = node->next;
  }
}

int edgeValidAt(xmlNode* node, int stepstart, int stepend)
{
  xmlNode* spells = xmlwGetChild(node, "spells");
  if (!spells)
    return 0;
  xmlNode* spell = xmlwGetChild(spells, "spell");
  if (!spell)
    return 0;
  int start = atoi((const char*) xmlGetProp(spell, (const xmlChar*) "start"));
  int end = atoi((const char*) xmlGetProp(spell, (const xmlChar*) "end"));
  if (start > stepend || stepstart > end)
    return 0;
  else
    return 1;
}

unsigned int gexfParseEdges(xmlNode* gexf, Graph* graph, int stepstart,
    int stepend)
{
  if (!gexf || !graph)
    return -1;
  xmlNode* xmlGraph = xmlwGetChild(gexf, "graph");
  if (!graph)
    return -1;
  xmlNode* xmledges = xmlwGetChild(xmlGraph, "edges");
  if (!xmledges)
    return -1;

  size_t i = 0;
  xmlNode* node = xmlwGetChild(xmledges, "edge");
  while (node)
  {
    if (edgeValidAt(node, stepstart, stepend)
        && xmlGetProp(node, (const xmlChar*) "id"))
    {
      gexfParseEdge(node, graph);
      i += 2;
    }
    node = node->next;
  }
  return i;
}

void gexfParseSetup(const char* in, xmlDoc** doc, xmlNode** root_element)
{
  if (!in)
  {
    printf("Invalid Input file pointer. Exit.\n");
    exit(EXIT_FAILURE);
  }

  /*parse the file and get the DOM */
  *doc = xmlReadFile(in, NULL, 256);

  if (*doc == NULL )
  {
    printf("error: could not parse file %s\n", in);
    exit(EXIT_FAILURE);
  }

  /*Get the root element node */
  *root_element = xmlDocGetRootElement(*doc);
}

void gexfParseCleanup(xmlDoc* doc)
{
  xmlFreeDoc(doc);
}

Vertices* gexfParseVerticesFromRoot(xmlNode* rootelem)
{
  unsigned int numvertices = xmlwGetNumNodes(rootelem);
  Vertices* vertices = newVertices(numvertices);
  vertices->numvertices = numvertices;
  gexfParseVertices(rootelem, vertices->vertexXLocs, vertices->vertexYLocs,
      vertices->vertexIds);
  return vertices;
}

Edges* gexfParseEdgesFromRoot(xmlNode* rootelem, Graph* graph, unsigned int maxedges)
{
  Edges* edges = newEdges(graph->vertices->numvertices);
  edges->maxedges = maxedges;
  graphSetEdgeSpaceForAllVertices(graph);
  gexfParseEdges(rootelem, graph, 0, INT_MAX);
  return edges;
}

Edges* gexfParseEdgesFromRootInInterval(xmlNode* rootelem, Graph* graph, unsigned int maxedges,
    int timestepStart, int timestepEnd)
{
  Edges* edges = newEdges(graph->vertices->numvertices);

  Edges* originalEdges = graph->edges;
  graph->edges = edges;
  graph->edges->maxedges = maxedges;

  graphSetEdgeSpaceForAllVertices(graph);
  gexfParseEdges(rootelem, graph, timestepStart, timestepEnd);

  graph->edges = originalEdges;

  return edges;
}

Graph* gexfParseFile(const char* in)
{
  Timer timer;
  startTimer(&timer);

  xmlDoc* doc;
  xmlNode* root_element;
  gexfParseSetup(in, &doc, &root_element);

  // Create graph data structure.
  Graph* g = newGraph(0);

  unsigned int numvertices = xmlwGetNumNodes(root_element);
  unsigned int maxEdges = xmlwGetMaxEdges(root_element, numvertices);
  g->vertices = gexfParseVerticesFromRoot(root_element);
  g->edges = gexfParseEdgesFromRoot(root_element, g, maxEdges);

  gexfParseCleanup(doc);

  /*
   * Free the global variables that may
   * have been allocated by the parser.
   */
  xmlCleanupParser();

  stopTimer(&timer);
  //printf("time: gexf parsing.\n");
  //printTimer(&timer);

  return g;
}

Vertices* gexfParseFileVertices(const char* in)
{
  xmlDoc* doc;
  xmlNode* rootelem;
  gexfParseSetup(in, &doc, &rootelem);
  Vertices* vertices = gexfParseVerticesFromRoot(rootelem);
  gexfParseCleanup(doc);
  return vertices;
}

Edges* gexfParseFileEdgesSomewhereInInterval(const char* in, Graph* graph,
    int stepstart, int stepend)
{
  xmlDoc* doc;
  xmlNode* rootelem;
  gexfParseSetup(in, &doc, &rootelem);
  unsigned int numvertices = xmlwGetNumNodes(rootelem);
  unsigned int maxedges = xmlwGetMaxEdges(rootelem, numvertices);
  Edges* edges = gexfParseEdgesFromRootInInterval(rootelem, graph, maxedges, stepstart,
      stepend);
  gexfParseCleanup(doc);
  return edges;
}

Edges** gexfParseFileEdgesAtSteps(const char* in, Graph* graph, int stepstart,
    int stepend, size_t* edgesLength)
{
  int interval = stepend - stepstart + 1;
  *edgesLength = interval + 1;
  if (interval < 1)
    return NULL ;
  Edges** edgeArray = (Edges**) calloc(*edgesLength, sizeof(Edges*));
  for (int i = 0; i < interval; i++)
  {
    edgeArray[i] = gexfParseFileEdgesSomewhereInInterval(in, graph,
        stepstart + i, stepstart + i);
  }
  edgeArray[interval] = gexfParseFileEdgesSomewhereInInterval(in, graph,
      stepstart, stepend);
  return edgeArray;
}
