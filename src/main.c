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
  int runForever = 0;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-i"))
    {
      // Input file param.
      inputFile = argv[++i];
    }
    else if (!strcmp(argv[i], "-I"))
    {
      runForever = 1;
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

  // Feedback to user.
  Graph* g = gexfParseFile(inputFile);

  // Printing
  //printGraph(g);

  // Computing.
  Timer timer;
  startTimer(&timer);
  fa2RunOnGraph(g, numTicks);

  unsigned int numgraphs = 10;
  Edges** edges = (Edges**) calloc(numgraphs, sizeof(Edges*));

  for (size_t i = 0; i < numgraphs; i++)
  {
    edges[i] = g->edges;
  }

  Vertices** verticesOut = (Vertices**) calloc(numgraphs, sizeof(Vertices*));

  for (size_t i = 0; i < numgraphs; i++)
  {
    verticesOut[i] = newVertices(g->vertices->numvertices);
  }

  fa2RunOnGraphInStream(g->vertices, edges, numgraphs, 10, verticesOut);
  stopTimer(&timer);
  //printf("time: total.\n");
  //printTimer(&timer);

  // Printing
  printGraph(g);

  free(g);
}

