/*!
  \file gexfparser.h
  Parses a gexf file to a graph.
 */
#ifndef GEXFPARSER_H
#define GEXFPARSER_H

#include "graph.h"

/*!
  Parses a gexf file to a graph.
  \param[out] g The outputted graph.
  \param[in] in The location of the gexf file to parse.
 */
void gexfParseFile(Graph* g, const char* in);

#endif
