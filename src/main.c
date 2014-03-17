
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

  for (size_t i = 0; i < argc; i++)
  {
    if (!strcmp(argv[i], "-i"))
    {
      // Input file param.
      inputFile = argv[i++];
    }
    else
    {
      printf("Unrecognized parameter: %s.\n", argv[i]);
      exit(EXIT_FAILURE);
    }
  }

  // Graph parsing.
  Graph* g = NULL;
  parseGexfFile(g, inputFile);

  for (size_t i = 0; i < numTicks; i++)
  {
    // Computing.
    // applyForceOnGraphEdge(g, myIncludedFunc);

    // Printing
    printGraph(g);
  } 
}

