#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "force-atlas-2.h"
#include "gexfparser.h"
#include "graph.h"
#include "timer.h"
#include "vector.h"
#include "vector-smoothening.h"
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

  float numvertices = graph->vertices->numvertices;

  // Transfer the vertex data to the gpu.
  graph->vertices->vertexXLocs =
   utilDataTransferHostToDevice(graph->vertices->vertexXLocs, numvertices *
       sizeof(float), 1);
  graph->vertices->vertexYLocs =
   utilDataTransferHostToDevice(graph->vertices->vertexYLocs, numvertices *
       sizeof(float), 1);

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

    graph->edges = edges[0];
    fa2RunOnGraph(graph, numTicks);

    // Add the final vertex positions to obtain the displacement.
    // utilVectorAdd(&speedvectors[0], graph->vertices->vertexXLocs,
    //     graph->vertices->numvertices);
    // utilVectorAdd(&speedvectors[graph->vertices->numvertices],
    //     graph->vertices->vertexYLocs, graph->vertices->numvertices);
    // vectorAverageShiftAndAdd(window, speedvectors);
  }

  float* averageSpeeds =
   vectorAverageNewVectorArray(graph->vertices->numvertices);
  vectorAverageComputeAverage(window,
      graph->vertices->numvertices, averageSpeeds);

  stopTimer(&timer);
  //printf("time: total.\n");
  //printTimer(&timer);

  // unsigned int* smootheningEdges;
  // unsigned int* smootheningNumEdges;
  // vectorSmootheningPrepareEdges(edges[edgesLength - 1]->edgeTargets,
  //     edges[edgesLength - 1]->numedges,
  //     edges[edgesLength - 1]->maxedges * graph->vertices->numvertices,
  //     graph->vertices->numvertices, &smootheningEdges, &smootheningNumEdges);
  // vectorSmootheningPrepareOutput(&smoothSpeedX, &smoothSpeedY,
  //     graph->vertices->numvertices);
  // vectorSmootheningRun(averageSpeedX, averageSpeedY,
  //     graph->vertices->numvertices, smootheningNumEdges, smootheningEdges, 10,
  //     0.5, smoothSpeedX, smoothSpeedY);
  // vectorSmootheningCleanEdges(smootheningEdges, smootheningNumEdges);

  // Printing
  printGraph(graph);
  freeGraph(graph);

  printf("Normal program exit.\n");
}

