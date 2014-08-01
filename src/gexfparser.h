/*!
 \file gexfparser.h
 Parses a gexf file to a graph.
 */
#ifndef GEXFPARSER_H
#define GEXFPARSER_H

#include "graph.h"
#include "libxml/parser.h"

/*!
 * Setups up the variable to look through the document.
 */
void gexfParseSetup(const char* in, xmlDoc** doc, xmlNode** root_element);

/*!
 * Cleans up the memory for looking through the document.
 */
void gexfParseCleanup(xmlDoc* doc);

/*!
 * Parses a gexf file to a graph.
 * \param[in] in The location of the gexf file to parse.
 * \return The new graph.
 */
Graph* gexfParseFile(const char* in);

/*!
 * Parses all vertices from the file and places them in a new Vertices struct.
 *
 * \param[in] in the full location of the gexf file.
 * \return A new vertices struct.
 */
Vertices* gexfParseFileVertices(const char* in);

/*!
 * Parses all edges from the gexf file that are valid at the given timestep.
 *
 * \param[in] in the gexf file to parse.
 * \param[in] timestep the timestap which is used to filter out all non-valid edges.
 * \return a new edges struct.
 */
Edges* gexfParseFileEdges(const char* in, int timestep);

/*!
 * Parses all edges from the gexf file stat are valid within the given interval.
 *
 * \param[in] in the gexf file to parse.
 * \param[in] timestart the begin of the time interval (inclusive)
 * \param[in] timeend the end of the time interval (inclusive)
 * \return an array of Edges. This array has size timeend - timestart.
 */
Edges** gexfParseFileEdgesInInterval(const char* in, int timestart,
    int timeend);

/*!
 * Looks through the document and finds the last timestep any edge is active.
 *
 * \param[in] in the gexf file to use.
 */
int gexfFindLastEdgeTime(const char* in);

#endif
