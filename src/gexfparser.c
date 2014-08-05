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

void gexfParseEdge(xmlNode* n, unsigned int* edgeSource,
    unsigned int* edgeTarget)
{
  if (!n || !edgeSource || !edgeTarget)
    return;
  *edgeSource = atoi((const char*) xmlGetProp(n, (const xmlChar*) "source"));
  *edgeTarget = atoi((const char*) xmlGetProp(n, (const xmlChar*) "target"));
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

unsigned int gexfParseEdges(xmlNode* gexf, unsigned int* edgeTargets,
    unsigned int* edgeSources, int stepstart, int stepend)
{
  if (!gexf || !edgeTargets || !edgeSources)
    return -1;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph)
    return -1;
  xmlNode* xmledges = xmlwGetChild(graph, "edges");
  if (!xmledges)
    return -1;

  size_t i = 0;
  xmlNode* node = xmlwGetChild(xmledges, "edge");
  while (node)
  {
    if (edgeValidAt(node, stepstart, stepend)
        && xmlGetProp(node, (const xmlChar*) "id"))
    {
      gexfParseEdge(node, &edgeSources[i], &edgeTargets[i]);
      i++;
      gexfParseEdge(node, &edgeTargets[i], &edgeSources[i]);
      i++;
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

Edges* gexfParseEdgesFromRoot(xmlNode* rootelem)
{
  unsigned int numedges = xmlwGetNumEdges(rootelem) * 2;
  Edges* edges = newEdges(numedges);
  edges->numedges = numedges;
  gexfParseEdges(rootelem, edges->edgeTargets, edges->edgeSources, 0, INT_MAX);
  return edges;
}

Edges* gexfParseEdgesFromRootInInterval(xmlNode* rootelem, int timestepStart,
    int timestepEnd)
{
  unsigned int numedges = xmlwGetNumEdges(rootelem) * 2;
  Edges* edges = newEdges(numedges);
  edges->numedges = numedges;
  numedges = gexfParseEdges(rootelem, edges->edgeTargets, edges->edgeSources,
      timestepStart, timestepEnd);
  edgesUpdateSize(edges, numedges);
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
  Graph* g = newGraph(0, 0);

  g->vertices = gexfParseVerticesFromRoot(root_element);
  g->edges = gexfParseEdgesFromRoot(root_element);

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

Edges* gexfParseFileEdgesSomewhereInInterval(const char* in, int stepstart,
    int stepend)
{
  xmlDoc* doc;
  xmlNode* rootelem;
  gexfParseSetup(in, &doc, &rootelem);
  Edges* edges = gexfParseEdgesFromRootInInterval(rootelem, stepstart, stepend);
  gexfParseCleanup(doc);
  return edges;
}

Edges** gexfParseFileEdgesAtSteps(const char* in, int stepstart, int stepend,
    size_t* edgesLength)
{
  int interval = stepend - stepstart + 1;
  *edgesLength = interval + 1;
  if (interval < 1)
    return NULL ;
  Edges** edgeArray = (Edges**) calloc(*edgesLength, sizeof(Edges*));
  for (int i = 0; i < interval; i++)
  {
    edgeArray[i] = gexfParseFileEdgesSomewhereInInterval(in, stepstart + i,
        stepstart + i);
  }
  edgeArray[interval] = gexfParseFileEdgesSomewhereInInterval(in, stepstart,
      stepend);
  return edgeArray;
}

int gexfFindLastEdgeTime(const char* in)
{
  xmlDoc* doc;
  xmlNode* rootelem;
  gexfParseSetup(in, &doc, &rootelem);
  int maxi = 0;

  xmlNode* graph = xmlwGetChild(rootelem, "graph");
  if (!graph)
    return -1;
  xmlNode* xmledges = xmlwGetChild(graph, "edges");
  if (!xmledges)
    return -1;

  xmlNode* node = xmlwGetChild(xmledges, "edge");
  while (node)
  {
    if (xmlGetProp(node, (const xmlChar*) "id"))
    {
      xmlNode* spells = xmlwGetChild(node, "spells");
      if (!spells)
        return -1;
      xmlNode* spell = xmlwGetChild(spells, "spell");
      if (!spell)
        return -1;
      int val = atoi((const char*) xmlGetProp(spell, (const xmlChar*) "end"));
      if (val > maxi)
        maxi = val;
    }

    node = node->next;
  }
  return maxi;
}
