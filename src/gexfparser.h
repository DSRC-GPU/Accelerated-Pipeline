/*!
  \file gexfparser.h
  Parses a gexf file to a graph.
 */
#ifndef GEXFPARSER_H
#define GEXFPARSER_H

#include "graph.h"

/*!
 * Parses a gexf file to a graph.
 * \param[in] in The location of the gexf file to parse.
 * \return The new graph.
 */
Graph* gexfParseFile(const char* in);

#endif
