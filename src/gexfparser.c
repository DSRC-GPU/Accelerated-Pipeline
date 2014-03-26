
#include "gexfparser.h"
#include "libxml/parser.h"

#include <stdlib.h>
#include <string.h>

#define NODE_START_X 100
#define NODE_START_Y 100

xmlNode* xmlwGetChild(xmlNode* node, const char* name)
{
  xmlNode* child = node->children;
  while (child)
  {
    if (!strcmp((const char*)child->name, name))
    {
      return child;
    }
    child = child->next;
  }
  return NULL;
}

unsigned int xmlwGetNumNodes(xmlNode* gexf)
{
  if (!gexf) return -1;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph) return -1;
  xmlNode* nodes = xmlwGetChild(graph, "nodes");
  if (!nodes) return -1;
  return atoi((const char*) xmlGetProp(nodes, (const xmlChar*) "count"));
}

unsigned int xmlwGetNumEdges(xmlNode* gexf)
{
  if (!gexf) return -1;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph) return -1;
  xmlNode* edges = xmlwGetChild(graph, "edges");
  if (!edges) return -1;
  return atoi((const char*) xmlGetProp(edges, (const xmlChar*) "count"));
}

void gexfParseVertex(xmlNode* n, float* vertexXLoc, float* vertexYLoc)
{
  if (!n || !vertexXLoc || !vertexYLoc) return;
  *vertexXLoc = NODE_START_X;
  *vertexYLoc = NODE_START_Y;
}

void gexfParseEdge(xmlNode* n, unsigned int* edgeSource,
    unsigned int* edgeTarget)
{
  if (!n || !edgeSource || !edgeTarget) return;
  *edgeSource = atoi((const char*) xmlGetProp(n, (const xmlChar*) "source"));
  *edgeTarget = atoi((const char*) xmlGetProp(n, (const xmlChar*) "target"));
}

void gexfParseVertices(xmlNode* gexf, float* vertexXLocs, float* vertexYLocs)
{
  if (!gexf || !vertexXLocs || !vertexYLocs) return;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph) return;
  xmlNode* nodes = xmlwGetChild(graph, "nodes");
  if (!nodes) return;

  size_t i = 0;
  xmlNode* node = nodes->children;
  while (node)
  {
    gexfParseVertex(node, &vertexXLocs[i], &vertexYLocs[i]);
    i++;
    node = node->next;
  }
}

void gexfParseEdges(xmlNode* gexf, unsigned int* edgeTargets,
    unsigned int* edgeSources)
{
  if (!gexf || !edgeTargets || !edgeSources) return;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph) return;
  xmlNode* xmledges = xmlwGetChild(graph, "edges");
  if (!xmledges) return;

  size_t i = 0;
  xmlNode* node = xmledges->children;
  while (node)
  {
    gexfParseEdge(node, &edgeSources[i], &edgeTargets[i]);
    i++;
    gexfParseEdge(node, &edgeTargets[i], &edgeSources[i]);
    i++;

    node = node->next;
  }
}

void gexfParseFile(Graph* g, const char* in)
{
  if (!g)
  {
    printf("Invalid Graph pointer. Exit.\n");
    exit(EXIT_FAILURE);
  }
  else if(!in)
  {
    printf("Invalid Input file pointer. Exit.\n");
    exit(EXIT_FAILURE);
  }

  xmlDoc *doc = NULL;
  xmlNode *root_element = NULL;

  /*parse the file and get the DOM */
  doc = xmlReadFile(in, NULL, 0);

  if (doc == NULL)
  {
    printf("error: could not parse file %s\n", in);
    exit(EXIT_FAILURE);
  }

  /*Get the root element node */
  root_element = xmlDocGetRootElement(doc);

  // Create graph data structure.
  unsigned int numNodes = xmlwGetNumNodes(root_element);
  unsigned int numEdges = xmlwGetNumEdges(root_element) * 2;

  g->numvertices = numNodes;
  g->numedges = numEdges;

  float* vertexXLoc = calloc(numNodes, sizeof(float));
  float* vertexYLoc = calloc(numNodes, sizeof(float));
  unsigned int* edgeStart = calloc(numEdges, sizeof(unsigned int));
  unsigned int* edgeEnd = calloc(numEdges, sizeof(unsigned int));

  g->vertexXLocs = vertexXLoc;
  g->vertexYLocs = vertexYLoc;
  g->edgeSources = edgeStart;
  g->edgeTargets = edgeEnd;

  gexfParseVertices(root_element, vertexXLoc, vertexYLoc);
  gexfParseEdges(root_element, edgeStart, edgeEnd);

  /*free the document */
  xmlFreeDoc(doc);

  /*
   *Free the global variables that may
   *have been allocated by the parser.
   */
  xmlCleanupParser();
}
