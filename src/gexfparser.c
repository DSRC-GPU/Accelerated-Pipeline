#include "gexfparser.h"
#include "libxml/parser.h"
#include "timer.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

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

void gexfParseEdges(xmlNode* gexf, unsigned int* edgeTargets,
    unsigned int* edgeSources)
{
  if (!gexf || !edgeTargets || !edgeSources)
    return;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph)
    return;
  xmlNode* xmledges = xmlwGetChild(graph, "edges");
  if (!xmledges)
    return;

  size_t i = 0;
  xmlNode* node = xmlwGetChild(xmledges, "edge");
  while (node)
  {
    if (xmlGetProp(node, (const xmlChar*) "id"))
    {
      gexfParseEdge(node, &edgeSources[i], &edgeTargets[i]);
      i++;
      gexfParseEdge(node, &edgeTargets[i], &edgeSources[i]);
      i++;
    }

    node = node->next;
  }
}

Graph* gexfParseFile(const char* in)
{
  Timer timer;

  startTimer(&timer);

  if (!in)
  {
    printf("Invalid Input file pointer. Exit.\n");
    exit(EXIT_FAILURE);
  }

  xmlDoc *doc = NULL;
  xmlNode *root_element = NULL;

  /*parse the file and get the DOM */
  doc = xmlReadFile(in, NULL, 256);

  if (doc == NULL )
  {
    printf("error: could not parse file %s\n", in);
    exit(EXIT_FAILURE);
  }

  /*Get the root element node */
  root_element = xmlDocGetRootElement(doc);

  // Create graph data structure.
  unsigned int numNodes = xmlwGetNumNodes(root_element);
  unsigned int numEdges = xmlwGetNumEdges(root_element) * 2;

  Graph* g = newGraph(numEdges, numNodes);

  g->vertices->numvertices = numNodes;
  g->edges->numedges = numEdges;

  float* vertexXLoc = (float*) calloc(numNodes, sizeof(float));
  float* vertexYLoc = (float*) calloc(numNodes, sizeof(float));
  unsigned int* edgeStart = (unsigned int*) calloc(numEdges,
      sizeof(unsigned int));
  unsigned int* edgeEnd = (unsigned int*) calloc(numEdges,
      sizeof(unsigned int));
  g->vertices->vertexIds = (int*) calloc(numNodes, sizeof(int));
  assert(g->vertices->vertexIds != NULL);

  g->vertices->vertexXLocs = vertexXLoc;
  g->vertices->vertexYLocs = vertexYLoc;
  g->edges->edgeSources = edgeStart;
  g->edges->edgeTargets = edgeEnd;

  gexfParseVertices(root_element, vertexXLoc, vertexYLoc,
      g->vertices->vertexIds);
  gexfParseEdges(root_element, edgeStart, edgeEnd);

  /*free the document */
  xmlFreeDoc(doc);

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
