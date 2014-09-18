
#ifndef PIPELINE_H_
#define PIPELINE_H_

#include "graph.h"

typedef struct PipelineData
{
  unsigned int numSpringEmbeddingIters;
  float phiFine;
  float phiCoarse;
  unsigned int phiFineRounds;
  unsigned int phiCoarseRounds;
  unsigned int windowSize;
} PipelineData;

void pipeline(const char* inputFile, PipelineData* data);

void pipelineSingleStep(Graph* graph, PipelineData* data);

#endif

