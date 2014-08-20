
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "force-atlas-2.h"
#include "gexfparser.h"
#include "graph.h"
#include "timer.h"
#include "vector.h"
#include "smoothening.h"
#include "pca.h"
#include "test-pca.h"
#include "test-util.h"
#include <stdio.h>
#include "vector-average.h"
#include "util.h"

int main(int argc, char* argv[])
{
  srand(time(NULL ));

  // Input parsing.
  const char* inputFile = NULL;
  unsigned int numTicks = 100;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-i"))
    {
      // Input file param.
      inputFile = argv[++i];
    }
    else if (!strcmp(argv[i], "-n"))
    {
      numTicks = atoi(argv[++i]);
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

  unsigned int numgraphs = 1;

  Vertices* vertices = gexfParseFileVertices(inputFile);
  graph->vertices = vertices;
  size_t edgesLength;
  Edges** edges = gexfParseFileEdgesAtSteps(inputFile, graph, 0, 199,
      &edgesLength);
  graph->edges = edges[0];

  float numvertices = graph->vertices->numvertices;
  float sizeEdgeArray = graph->edges->maxedges * numvertices;

  // Transfer the vertex data to the gpu.
  graph->vertices->vertexXLocs = (float*)
   utilDataTransferHostToDevice(graph->vertices->vertexXLocs, numvertices *
       sizeof(float), 1);
  graph->vertices->vertexYLocs = (float*)
   utilDataTransferHostToDevice(graph->vertices->vertexYLocs, numvertices *
       sizeof(float), 1);

  // Transfer the edge data to the gpu.
  graph->edges->numedges = (unsigned int*) utilDataTransferHostToDevice(graph->edges->numedges,
      numvertices * sizeof(unsigned int), 1);
  graph->edges->edgeTargets = (unsigned int*)
   utilDataTransferHostToDevice(graph->edges->edgeTargets,
      sizeEdgeArray * sizeof(unsigned int), 1);

  float** window = vectorAverageNewWindow();

  // Computing.
  Timer timer;
  startTimer(&timer);

  for (size_t i = 0; i < WINDOW_SIZE; i++)
  {
    // Create new speed vectors and set them to the negative value of the vertex
    // stars positions.
    float* speedvectors =
     vectorAverageNewVectorArray(graph->vertices->numvertices);
    utilVectorSetByScalar(speedvectors, 0, graph->vertices->numvertices * 2);
    utilVectorAdd(&speedvectors[0], graph->vertices->vertexXLocs,
        graph->vertices->numvertices);
    utilVectorAdd(&speedvectors[graph->vertices->numvertices],
        graph->vertices->vertexYLocs, graph->vertices->numvertices);
    utilVectorMultiplyByScalar(speedvectors, -1, graph->vertices->numvertices *
        2);

    fa2RunOnGraph(graph, numTicks);

    // Add the final vertex positions to obtain the displacement.
    utilVectorAdd(&speedvectors[0], graph->vertices->vertexXLocs,
        graph->vertices->numvertices);
    utilVectorAdd(&speedvectors[graph->vertices->numvertices],
        graph->vertices->vertexYLocs, graph->vertices->numvertices);
    vectorAverageShiftAndAdd(window, speedvectors);

    DEBUG_PRINT_DEVICE(speedvectors, numvertices * 2);
  }

  float* averageSpeeds =
   vectorAverageNewVectorArray(graph->vertices->numvertices);
  vectorAverageComputeAverage(window,
      graph->vertices->numvertices, averageSpeeds);

  DEBUG_PRINT_DEVICE(averageSpeeds, numvertices * 2);

  stopTimer(&timer);
  printf("time: total.\n");
  printTimer(&timer);

  float* projectedData = utilAllocateData(numvertices * 2 * sizeof(float));
  pca(averageSpeeds, 2, numvertices, projectedData);

  DEBUG_PRINT_DEVICE(projectedData, numvertices * 2);

  float* smoothFineValues = utilAllocateData(numvertices * sizeof(float));
  float* smoothCoarseValues = utilAllocateData(numvertices * sizeof(float));
  unsigned int* smootheningEdges;
  unsigned int* smootheningNumEdges;
  smootheningPrepareEdges(edges[edgesLength - 1]->edgeTargets,
      edges[edgesLength - 1]->numedges,
      edges[edgesLength - 1]->maxedges * graph->vertices->numvertices,
      graph->vertices->numvertices, &smootheningEdges, &smootheningNumEdges);
  smootheningPrepareOutput(&smoothFineValues, graph->vertices->numvertices);
  smootheningPrepareOutput(&smoothCoarseValues, graph->vertices->numvertices);
  smootheningRun(projectedData,
      graph->vertices->numvertices, smootheningNumEdges, smootheningEdges, 10,
      0, smoothFineValues);
  smootheningRun(projectedData,
      graph->vertices->numvertices, smootheningNumEdges, smootheningEdges, 10,
      1, smoothCoarseValues);
  smootheningCleanEdges(smootheningEdges, smootheningNumEdges);

  // TODO Free memory with the util functions.

  printf("Normal program exit.\n");
}

