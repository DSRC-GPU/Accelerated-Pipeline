
#include "gexfparser.h"
#include "libxml/parser.h"

#include <stdlib.h>
#include <string.h>

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

void gexfParseVertex(xmlNode* n, Vertex* v)
{
  if (!n || !v) return;
  v->id = atoi((const char*) xmlGetProp(n, (const xmlChar*)  "id"));
  v->label = atoi((const char*) xmlGetProp(n, (const xmlChar*) "label"));
  xmlNode* spells = xmlwGetChild(n, "spells");
  if (spells)
  {
    xmlNode* spell = xmlwGetChild(spells, "spell");
    if (spell)
    {
      v->start = atoi((const char*) xmlGetProp(spell,
            (const xmlChar*) "start"));
      v->end = atoi((const char*) xmlGetProp(spell, (const xmlChar*) "end"));
    }
  }
  v->loc.x = 0;
  v->loc.y = 0;
  v->force.x = 0;
  v->force.y = 0;
  v->neighbourIndex = -1;
}

void gexfParseEdge(xmlNode* n, Edge* e)
{
  if (!n || !e) return;
  e->startVertex = atoi((const char*) xmlGetProp(n, (const xmlChar*) "source"));
  e->endVertex = atoi((const char*) xmlGetProp(n, (const xmlChar*) "target"));
  xmlNode* spells = xmlwGetChild(n, "spells");
  if (spells)
  {
    xmlNode* spell = xmlwGetChild(spells, "spell");
    if (spell)
    {
      e->start = atoi((const char*) xmlGetProp(spell, (const xmlChar*) "start"));
      e->end = atoi((const char*) xmlGetProp(spell, (const xmlChar*) "end"));
    }
  }
}

void gexfParseVertices(xmlNode* gexf, Vertex* vertices)
{
  if (!gexf || !vertices) return;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph) return;
  xmlNode* nodes = xmlwGetChild(graph, "nodes");
  if (!nodes) return;

  size_t i = 0;
  xmlNode* node = nodes->children;
  while (node)
  {
    gexfParseVertex(node, &vertices[i++]);
    node = node->next;
  }
}

void gexfParseEdges(xmlNode* gexf, Edge* edges)
{
  if (!gexf || !edges) return;
  xmlNode* graph = xmlwGetChild(gexf, "graph");
  if (!graph) return;
  xmlNode* xmledges = xmlwGetChild(graph, "edges");
  if (!xmledges) return;

  size_t i = 0;
  xmlNode* node = xmledges->children;
  while (node)
  {
    gexfParseEdge(node, &edges[i++]);
    node = node->next;
  }
}

void connectEdgesVertices(Graph* g)
{
  qsort(g->edges, g->numedges, sizeof(Edge), compare_edges);
  for (size_t i = 0; i < g->numedges; i++)
  {
    unsigned int edgeIndex = g->numedges - (i + 1);
    g->vertices[g->edges[edgeIndex].startVertex].neighbourIndex = edgeIndex;
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
  unsigned int numEdges = xmlwGetNumEdges(root_element);

  g->numvertices = numNodes;
  g->numedges = numEdges;

  Vertex* vertices = calloc(numNodes, sizeof(Vertex));
  Edge* edges = calloc(numEdges, sizeof(Edge));

  g->vertices = vertices;
  g->edges = edges;

  gexfParseVertices(root_element, vertices);
  gexfParseEdges(root_element, edges);
  connectEdgesVertices(g);

  /*free the document */
  xmlFreeDoc(doc);

  /*
   *Free the global variables that may
   *have been allocated by the parser.
   */
  xmlCleanupParser();
}
