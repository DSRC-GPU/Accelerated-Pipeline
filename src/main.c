
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "force-atlas-2.h"
#include "gexfparser.h"
#include "graph.h"
#include "vector.h"

int main(int argc, char* argv[])
{
  srand(time(NULL));

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
    // Printing
    printGraph(g);

    // Computing.
    fa2RunOnGraph(g);

    i++;
  } 

  free(g);
}

