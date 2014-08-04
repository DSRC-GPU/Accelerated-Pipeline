#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "force-atlas-2.h"
#include "gexfparser.h"
#include "graph.h"
#include "timer.h"
#include "vector.h"

int main(int argc, char* argv[])
{
  srand(time(NULL ));

  // Input parsing.
  const char* inputFile = NULL;
  unsigned int numTicks = 100;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-i"))
    {
      // Input file param.
      inputFile = argv[++i];
    }
    else if (!strcmp(argv[i], "-n"))
    {
      numTicks = atoi(argv[++i]);
    }
    else
    {
      printf("Unrecognized parameter: %s.\n", argv[i]);
      //   exit(EXIT_FAILURE);
    }
  }

  // Input checking.
  if (!inputFile)
  {
    printf("No input file specified. Exit.\n");
    exit(EXIT_FAILURE);
  }

  unsigned int numgraphs = 1;
  Vertices* vertices = gexfParseFileVertices(inputFile);
  //Edges* edges = gexfParseFileEdges(inputFile, 0);
  Edges** edges = gexfParseFileEdgesInInterval(inputFile, 0, 199);

  Graph* testgraph = (Graph*) calloc(1, sizeof(Graph));
  testgraph->vertices = vertices;
  //testgraph->edges = edges;

  Vertices** verticesOut = (Vertices**) calloc(numgraphs, sizeof(Vertices*));
  for (size_t i = 0; i < numgraphs; i++)
  {
    verticesOut[i] = newVertices(vertices->numvertices);
    verticesOut[i]->numvertices = vertices->numvertices;
  }

  // Computing.
  Timer timer;
  startTimer(&timer);
  fa2RunOnGraphInStream(testgraph->vertices, edges, numgraphs,
      numTicks, verticesOut);
  stopTimer(&timer);
  //printf("time: total.\n");
  //printTimer(&timer);

  // Printing
  testgraph->vertices = verticesOut[numgraphs - 1];
  printGraph(testgraph);
}

