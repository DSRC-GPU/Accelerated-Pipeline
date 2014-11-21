
#ifndef GRAPHPARSER_H
#define GRAPHPARSER_H

#include "graph.h"

/*!
 * Returns parsed graph. User has to free memory when done.
 * \param[in] filename Name of the graph file.
 * \return parsed graph.
 */
Graph* graphParse(const char* filename);

#endif
