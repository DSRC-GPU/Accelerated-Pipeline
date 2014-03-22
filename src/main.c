
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "gexfparser.h"
#include "graph.h"
#include "forces.h"
#include "force-atlas-2.h"



int main(int argc, char* argv[])
{
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
    else
    {
      printf("Unrecognized parameter: %s.\n", argv[i]);
      exit(EXIT_FAILURE);
    }
  }

  // Input checking.
  if (!inputFile)
  {
    printf("No input file specified. Exit.\n");
    exit(EXIT_FAILURE);
  }

  // Feedback to user.
  printf("Using input file %s.\n", inputFile);
  if (runForever)
    printf("Running simulation indefinately.\n");
  else
    printf("Running simulation for %d ticks.\n", numTicks);

  // Graph parsing.
  printf("Parsing graph...");
  Graph* g = calloc(1, sizeof(Graph));
  gexfParseFile(g, inputFile);
  printf(" done!\n");

  printf("Graph nodes: %d, Graph edges: %d.\n", g->numvertices, g->numedges);

  size_t i = 0;
  while (i < numTicks || runForever)
  {
    // Computing.
    updateForcesOnGraph(g, FA2_NUMFORCES, FA2_FORCES);
    updateSpeedOnGraph(g);
    updateLocationOnGraph(g);
    resetForcesOnGraph(g);

    // Printing
    printGraph(g);

    i++;
  } 

  free(g);
}

