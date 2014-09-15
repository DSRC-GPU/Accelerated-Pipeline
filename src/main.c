
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
#include "break-edges.h"
#include "connected-component.h"

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
  unsigned int numTicks = 100;
  float phiFine = 0.01;
  float phiCoarse = 0.3;
  unsigned int phiFineRounds = 200;
  unsigned int phiCoarseRounds = 40;
  unsigned int windowSize = 30;

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
    else if (!strcmp(argv[i], argPhiFine))
    {
      phiFine = atof(argv[++i]);
    }
    else if (!strcmp(argv[i], argPhiCoarse))
    {
      phiCoarse = atof(argv[++i]);
    }
    else if (!strcmp(argv[i], argPhiFineRounds))
    {
      phiFineRounds = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], argPhiCoarseRounds))
    {
      phiCoarseRounds = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], argWindowSize))
    {
      windowSize = atoi(argv[++i]);
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
  FILE* outputFile = fopen("out", "w");

  Graph* graph = (Graph*) calloc(1, sizeof(Graph));

  Vertices* vertices = gexfParseFileVertices(inputFile);
  graph->vertices = vertices;
  Edges* edges = gexfParseFileEdgesSomewhereInInterval(inputFile, graph, 0,
      WINDOW_SIZE);
  graph->edges = edges;

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

  // Declare variables for loop.
  float* smoothFineValues = (float*) utilAllocateData(numvertices * sizeof(float));
  float* smoothCoarseValues = (float*) utilAllocateData(numvertices * sizeof(float));
  unsigned int* vertexlabels = (unsigned int*) utilAllocateData(numvertices
      * sizeof(unsigned int));
  float* projectedData = (float*) utilAllocateData(numvertices * 2 * sizeof(float));
  float* averageSpeeds =
   vectorAverageNewVectorArray(graph->vertices->numvertices);

  // Never stop. Break manually when end of file is reached.
  for (size_t graphIteration = 0; graphIteration >= 0; graphIteration++)
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
    vectorAverageComputeAverage(window,
        graph->vertices->numvertices, averageSpeeds);

    stopTimer(&timer);
    printf("time: total.\n");
    printTimer(&timer);

    pca(averageSpeeds, 2, numvertices, projectedData);

    smootheningRun(projectedData,
        graph->vertices->numvertices, graph->edges->numedges,
        graph->edges->edgeTargets, phiFineRounds, phiFine, smoothFineValues);
    smootheningRun(projectedData,
        graph->vertices->numvertices, graph->edges->numedges,
        graph->edges->edgeTargets, phiCoarseRounds, phiCoarse, smoothCoarseValues);

    // TODO Free memory with the util functions.

    breakEdges(graph->vertices->numvertices, smoothFineValues, smoothCoarseValues,
        graph->edges->numedges, graph->edges->edgeTargets);

    connectedComponent(graph->vertices->numvertices, graph->edges->numedges,
        graph->edges->edgeTargets, vertexlabels);

    unsigned int* h_vertexlabels = (unsigned int*) utilDataTransferDeviceToHost(vertexlabels,
        numvertices * sizeof(unsigned int), 1);

    for (size_t i = 0; i < numvertices; i++)
    {
      fprintf(outputFile, "%u ", h_vertexlabels[i]);
    }
    fprintf(outputFile, "\n");

    // Load the next set of edges.
    utilFreeDeviceData(edges->numedges);
    utilFreeDeviceData(edges->edgeTargets);
    edges = gexfParseFileEdgesSomewhereInInterval(inputFile, graph,
        graphIteration, graphIteration + WINDOW_SIZE);
    if (!edges->maxedges)
    {
      break;
    } else {
      printf("Window is at %lu to %lu, maxedges: %u\n", graphIteration,
          graphIteration + WINDOW_SIZE, edges->maxedges);

      // Clean previous edges
      free(graph->edges);
      graph->edges = edges;
      
      // Transfer the edge data to the gpu.
      graph->edges->numedges = (unsigned int*) utilDataTransferHostToDevice(graph->edges->numedges,
          numvertices * sizeof(unsigned int), 1);
      graph->edges->edgeTargets = (unsigned int*)
       utilDataTransferHostToDevice(graph->edges->edgeTargets,
          sizeEdgeArray * sizeof(unsigned int), 1);
    }
  }

  // Clean up
  utilFreeDeviceData(graph->vertices->vertexXLocs);
  utilFreeDeviceData(graph->vertices->vertexYLocs);
  utilFreeDeviceData(edges->numedges);
  utilFreeDeviceData(edges->edgeTargets);
  free(graph->vertices);
  free(edges);

  free(graph);
  utilFreeDeviceData(smoothFineValues);
  utilFreeDeviceData(smoothCoarseValues);
  utilFreeDeviceData(projectedData);
  utilFreeDeviceData(vertexlabels);
  vectorAverageFreeVectorArray(averageSpeeds);
  vectorAverageFreeWindow(window);
  
  fclose(outputFile);
  printf("Normal program exit.\n");
}

