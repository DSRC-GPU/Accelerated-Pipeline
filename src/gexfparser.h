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
 * Parses all edges from the gexf file that are valid at some point in the given interval.
 *
 * \param[in] in the gexf file to parse.
 * \param[in] stepstart the begin of the interval.
 * \param[in] stepend the end of the interval
 * \return a new edges struct.
 */
Edges* gexfParseFileEdgesSomewhereInInterval(const char* in, int stepstart,
    int stepend);

/*!
 * Parses all edges from the gexf file stat are valid within the given interval.
 * The Edges* at index k (k <= timeend - timestart) are edges that are valid at timestep k - 1.
 * These can be used for the edges for every timestep within the time window.
 *
 * The Edges* at index k (k = timeend - timestart + 1) are edges that are valid
 * somewhere in the interval [timestart, timeend].
 * These can be used for the smoothening of the average speed vectors.
 *
 * \param[in] in the gexf file to parse.
 * \param[in] graph the graph that contains the vertices of the graph.
 * \param[in] timestart the begin of the time interval (inclusive)
 * \param[in] timeend the end of the time interval (inclusive)
 * \param[out] edgesLength the length of the returned edges array.
 * In case you are unable to calculate.
 * \return an array of Edges. This array has size timeend - timestart + 1.
 */
Edges** gexfParseFileEdgesAtSteps(const char* in, Graph* graph, int timestart,
    int timeend, size_t* edgesLength);

/*!
 * Looks through the document and finds the last timestep any edge is active.
 *
 * \param[in] in the gexf file to use.
 */
int gexfFindLastEdgeTime(const char* in);

#endif
