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
#include <stdio.h>

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
      // exit(EXIT_FAILURE);
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
  size_t edgesLength;
  Edges** edges = gexfParseFileEdgesAtSteps(inputFile, 0, 199, &edgesLength);

  Graph* graph = (Graph*) calloc(1, sizeof(Graph));
  graph->vertices = vertices;

  float* averageSpeedX = NULL;
  float* averageSpeedY = NULL;

  // Computing.
  Timer timer;
  startTimer(&timer);
  fa2RunOnGraphInStream(graph->vertices, edges, numgraphs, numTicks,
      &averageSpeedX, &averageSpeedY);
  stopTimer(&timer);
  //printf("time: total.\n");
  //printTimer(&timer);

  // Printing
  printGraph(graph);

  freeGraph(graph);
}

