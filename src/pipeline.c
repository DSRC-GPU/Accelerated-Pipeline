
#include "pipeline.h"

#include <stdlib.h>
#include <stdio.h>

#include "gexfparser.h"
#include "vector-average.h"
#include "util.h"
#include "force-atlas-2.h"
#include "pca.h"
#include "smoothening.h"
#include "break-edges.h"
#include "connected-component.h"
#include "cuda-timer.h"

void pipeline(const char* inputFile, PipelineData* data)
{
  Graph* graph = (Graph*) calloc(1, sizeof(Graph));
  graph->vertices = gexfParseFileVertices(inputFile);
  graph->edges = gexfParseFileEdgesSomewhereInInterval(inputFile, graph, 0,
      WINDOW_SIZE);

  float numvertices = graph->vertices->numvertices;

  CudaTimer datatransferTimer;
  startCudaTimer(&datatransferTimer);

  // Transfer the vertex data to the gpu.
  graph->vertices->vertexXLocs = (float*)
   utilDataTransferHostToDevice(graph->vertices->vertexXLocs, numvertices *
       sizeof(float), 1);
  graph->vertices->vertexYLocs = (float*)
   utilDataTransferHostToDevice(graph->vertices->vertexYLocs, numvertices *
       sizeof(float), 1);

  stopCudaTimer(&datatransferTimer);
  printCudaTimer(&datatransferTimer, "copying vertices to device.");

  CudaTimer iterationTimer, edgesToDeviceTimer;

  unsigned int iter = 1;
  while (graph->edges->maxedges > 0)
  {

    float sizeEdgeArray = graph->edges->maxedges * numvertices;

    startCudaTimer(&edgesToDeviceTimer);

    // Transfer the edge data to the gpu.
    graph->edges->numedges = (unsigned int*) utilDataTransferHostToDevice(graph->edges->numedges,
        numvertices * sizeof(unsigned int), 1);
    graph->edges->edgeTargets = (unsigned int*)
     utilDataTransferHostToDevice(graph->edges->edgeTargets,
        sizeEdgeArray * sizeof(unsigned int), 1);

    stopCudaTimer(&edgesToDeviceTimer);
    printCudaTimer(&edgesToDeviceTimer, "copying edges to device.");

    startCudaTimer(&iterationTimer);

    pipelineSingleStep(graph, data);

    stopCudaTimer(&iterationTimer);
    printCudaTimer(&iterationTimer, "pipeline iteration.");

    utilFreeDeviceData(graph->edges->numedges);
    utilFreeDeviceData(graph->edges->edgeTargets);
    free(graph->edges);
    graph->edges = gexfParseFileEdgesSomewhereInInterval(inputFile, graph, iter,
        WINDOW_SIZE + iter);

    iter++;
  }

  // Clean up
  utilFreeDeviceData(graph->vertices->vertexXLocs);
  utilFreeDeviceData(graph->vertices->vertexYLocs);
  free(graph->vertices);
  free(graph->edges->numedges);
  free(graph->edges->edgeTargets);
  free(graph->edges);
  free(graph);
}

void pipelineSingleStep(Graph* graph, PipelineData* data)
{
    // Create new speed vectors and set them to the negative value of the vertex
    // stars positions.
#ifdef DEBUG
    DEBUG_PRINT("LOCATIONS\n");
    utilPrintDeviceArray(graph->vertices->vertexXLocs,
        graph->vertices->numvertices);
#endif

    float numvertices = graph->vertices->numvertices;

    // Declare variables loop.
    static float** window = vectorAverageNewWindow();
    static float* smoothFineValues = (float*) utilAllocateData(numvertices * sizeof(float));
    static float* smoothCoarseValues = (float*) utilAllocateData(numvertices * sizeof(float));
    static unsigned int* vertexlabels = (unsigned int*) utilAllocateData(numvertices
        * sizeof(unsigned int));
    static float* projectedData = (float*) utilAllocateData(numvertices * 2 * sizeof(float));
    static float* averageSpeeds =
     vectorAverageNewVectorArray(graph->vertices->numvertices);

    static FILE* outputFile = fopen("out", "w");

    static CudaTimer speedvectorInitTimer, springEmbeddingTimer,
                     speedvectorFinishTimer, pcaTimer, smootheningFineTimer,
                     smootheningCoarseTimer, breakEdgesTimer,
                     connectedComponentTimer, verticesToHostTimer;

    float* speedvectors =
     vectorAverageNewVectorArray(graph->vertices->numvertices);

    startCudaTimer(&speedvectorInitTimer);

    utilVectorSetByScalar(speedvectors, 0, graph->vertices->numvertices * 2);
    utilVectorAdd(&speedvectors[0], graph->vertices->vertexXLocs,
        graph->vertices->numvertices);
    utilVectorAdd(&speedvectors[graph->vertices->numvertices],
        graph->vertices->vertexYLocs, graph->vertices->numvertices);
    utilVectorMultiplyByScalar(speedvectors, -1, graph->vertices->numvertices *
        2);

    stopCudaTimer(&speedvectorInitTimer);
    printCudaTimer(&speedvectorInitTimer, "setting up speed vectors");
    startCudaTimer(&springEmbeddingTimer);

    fa2RunOnGraph(graph, data->numSpringEmbeddingIters);

    stopCudaTimer(&springEmbeddingTimer);
    printCudaTimer(&springEmbeddingTimer, "running spring-embedding");
    startCudaTimer(&speedvectorFinishTimer);

    // Add the final vertex positions to obtain the displacement.
    utilVectorAdd(&speedvectors[0], graph->vertices->vertexXLocs,
        graph->vertices->numvertices);
    utilVectorAdd(&speedvectors[graph->vertices->numvertices],
        graph->vertices->vertexYLocs, graph->vertices->numvertices);
    vectorAverageShiftAndAdd(window, speedvectors);
    vectorAverageComputeAverage(window,
        graph->vertices->numvertices, averageSpeeds);

    stopCudaTimer(&speedvectorFinishTimer);
    printCudaTimer(&speedvectorFinishTimer, "finishing speed vectors");
    startCudaTimer(&pcaTimer);

    pca(averageSpeeds, 2, numvertices, projectedData);

    stopCudaTimer(&pcaTimer);
    printCudaTimer(&pcaTimer, "pca");
    startCudaTimer(&smootheningFineTimer);

    smootheningRun(projectedData,
        graph->vertices->numvertices, graph->edges->numedges,
        graph->edges->edgeTargets, data->phiFineRounds, data->phiFine, smoothFineValues);

    stopCudaTimer(&smootheningFineTimer);
    printCudaTimer(&smootheningFineTimer, "smoothening fine");
    startCudaTimer(&smootheningCoarseTimer);

    smootheningRun(projectedData,
        graph->vertices->numvertices, graph->edges->numedges,
        graph->edges->edgeTargets, data->phiCoarseRounds, data->phiCoarse, smoothCoarseValues);

    stopCudaTimer(&smootheningCoarseTimer);
    printCudaTimer(&smootheningCoarseTimer, "smoothening coarse");
    startCudaTimer(&breakEdgesTimer);

    breakEdges(graph->vertices->numvertices, smoothFineValues, smoothCoarseValues,
        graph->edges->numedges, graph->edges->edgeTargets);

    stopCudaTimer(&breakEdgesTimer);
    printCudaTimer(&breakEdgesTimer, "breaking edges");
    startCudaTimer(&connectedComponentTimer);

    connectedComponent(graph->vertices->numvertices, graph->edges->numedges,
        graph->edges->edgeTargets, vertexlabels);

    stopCudaTimer(&connectedComponentTimer);
    printCudaTimer(&connectedComponentTimer, "connected component");
    startCudaTimer(&verticesToHostTimer);

    unsigned int* h_vertexlabels = (unsigned int*) utilDataTransferDeviceToHost(vertexlabels,
        numvertices * sizeof(unsigned int), 0);

    stopCudaTimer(&verticesToHostTimer);
    printCudaTimer(&verticesToHostTimer, "copying vertex labels to host");

    for (size_t i = 0; i < numvertices; i++)
    {
      fprintf(outputFile, "%u ", h_vertexlabels[i]);
    }
    fprintf(outputFile, "\n");

    // utilFreeDeviceData(speedvectors);
    // utilFreeDeviceData(smoothFineValues);
    // utilFreeDeviceData(smoothCoarseValues);
    // utilFreeDeviceData(projectedData);
    // utilFreeDeviceData(vertexlabels);
    // vectorAverageFreeVectorArray(averageSpeeds);
    // vectorAverageFreeWindow(window);
}

