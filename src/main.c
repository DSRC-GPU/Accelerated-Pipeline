
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>

#include "pipeline.h"

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

  PipelineData data;
  data.numSpringEmbeddingIters = 300;
  data.phiFine = 0.01;
  data.phiCoarse = 0.3;
  data.phiFineRounds = 200;
  data.phiCoarseRounds = 40;
  data.windowSize = 30;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-i"))
    {
      // Input file param.
      inputFile = argv[++i];
    }
    else if (!strcmp(argv[i], "-n"))
    {
      data.numSpringEmbeddingIters = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], argPhiFine))
    {
      data.phiFine = atof(argv[++i]);
    }
    else if (!strcmp(argv[i], argPhiCoarse))
    {
      data.phiCoarse = atof(argv[++i]);
    }
    else if (!strcmp(argv[i], argPhiFineRounds))
    {
      data.phiFineRounds = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], argPhiCoarseRounds))
    {
      data.phiCoarseRounds = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], argWindowSize))
    {
      data.windowSize = atoi(argv[++i]);
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
  
  pipeline(inputFile, &data); 

  printf("Normal program exit.\n");
}

