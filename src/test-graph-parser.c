/*!
 * This program only runs the spring embedding algorithm for a number of
 * iterations.
 *
 * Used to evaluate the performance of the spring embedding without having to
 * run the complete pipeline.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>

#include "gexfparser.h"
#include "graphparser.h"
#include "pipeline.h"
#include "timer.h"
#include "util.h"
#include "force-atlas-2.h"
#include "vector-average.h"

const char* argPhiFine = "--phi-fine";
const char* argPhiCoarse = "--phi-coarse";
const char* argPhiFineRounds = "--phi-fine-rounds";
const char* argPhiCoarseRounds = "--phi-coarse-rounds";
const char* argWindowSize = "--window-size";

int main(int argc, char* argv[])
{
  srand(time(NULL ));

  // Input parsing.
  const char* inputFile = NULL;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-i"))
    {
      // Input file param.
      inputFile = argv[++i];
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
  
  Graph* graph = graphParse(inputFile);
  printGraph(graph);
  printGraphEdges(graph);
  freeGraph(graph);
  
  return 0;

  printf("Normal program exit.\n");
}

