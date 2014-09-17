
#ifndef PIPELINE_H_
#define PIPELINE_H_

typedef struct PipelineData
{
  unsigned int numSpringEmbeddingIters;
  float phiFine;
  float phiCoarse;
  unsigned int phiFineRounds;
  unsigned int phiCoarseRounds;
  unsigned int windowSize;
} PipelineData;

void pipeline(Graph* graph, PipelineData* data);

void pipelineSingleStep(Graph* graph, PipelineData* data);

#endif

