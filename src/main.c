
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "gexfparser.h"
#include "graph.h"

int main(int argc, char* argv[])
{
  // Input handling.
  char* inputFile = "";
  unsigned int numTicks = 100;
  int runForever = 0;

  for (size_t i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-i"))
    {
      // Input file param.
      inputFile = argv[i++];
    }
    if (!strcmp(argv[i], "-I"))
    {
      runForever = 1;
    }
    else
    {
      printf("Unrecognized parameter: %s.\n", argv[i]);
      exit(EXIT_FAILURE);
    }
  }

  // Graph parsing.
  Graph* g = NULL;
  gexfParseFile(g, inputFile);

  size_t i = 0;
  while (i < numTicks || runForever)
  {
    // Computing.
    // applyForceOnGraphEdge(g, myIncludedFunc);

    // Printing
    printGraph(g);

    numTicks++;
  } 
}

