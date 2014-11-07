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
  
  Graph* graph = (Graph*) calloc(1, sizeof(Graph));
  graph->vertices = gexfParseFileVertices(inputFile);
  graph->edges = gexfParseFileEdgesSomewhereInInterval(inputFile, graph, 0,
      WINDOW_SIZE);

  float numvertices = graph->vertices->numvertices;

  Timer* datatransferTimer = timerNew();
  startTimer(datatransferTimer);

  // Transfer the vertex data to the gpu.
  graph->vertices->vertexXLocs = (float*)
   utilDataTransferHostToDevice(graph->vertices->vertexXLocs, numvertices *
       sizeof(float), 1);
  graph->vertices->vertexYLocs = (float*)
   utilDataTransferHostToDevice(graph->vertices->vertexYLocs, numvertices *
       sizeof(float), 1);

  float sizeEdgeArray = graph->edges->maxedges * numvertices;

  // Transfer the edge data to the gpu.
  graph->edges->numedges = (unsigned int*) utilDataTransferHostToDevice(graph->edges->numedges,
      numvertices * sizeof(unsigned int), 1);
  graph->edges->edgeTargets = (unsigned int*)
   utilDataTransferHostToDevice(graph->edges->edgeTargets,
      sizeEdgeArray * sizeof(unsigned int), 1);

  fa2RunOnGraph(graph, data->numSpringEmbeddingIters);

  // Clean up edges.
  utilFreeDeviceData(graph->edges->numedges);
  utilFreeDeviceData(graph->edges->edgeTargets);
  free(graph->edges);

  // Clean up rest.
  utilFreeDeviceData(graph->vertices->vertexXLocs);
  utilFreeDeviceData(graph->vertices->vertexYLocs);
  free(graph->vertices);
  free(graph->edges->numedges);
  free(graph->edges->edgeTargets);
  free(graph->edges);
  free(graph);


  printf("Normal program exit.\n");
}

