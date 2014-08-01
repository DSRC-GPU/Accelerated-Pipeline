/*!
 * \file force-atlas-2.cu
 * A parallel implementation of the Force Atlas 2 spring embedding algorithm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "force-atlas-2.h"
#include "math.h"
#include "cuda-timer.h"
#include "cuda-stream.h"
#include "vector.h"

/*!
 * The number of thread in each thread block.
 */
#define BLOCK_SIZE 64

/*!
 * Updates the current force on each vertex with the current gravity.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] vxLocs Array that holds the x location of all vertices.
 * \param[in] vyLocs Array that holds the y location of all vertices.
 * \param[out] forceX Pointer to the x force on the vertex that belongs to this
 *    thread.
 * \param[out] forceY Pointer to the y force on the vertex that belongs to this
 *    thread.
 * \param[in] deg Array holding the out degree values for each vertex.
 */
__device__ void fa2Gravity(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg);

/*!
 * Updates the current force on each vertex with the current repulsion.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] vxLocs Array that holds the x location of all vertices.
 * \param[in] vyLocs Array that holds the y location of all vertices.
 * \param[out] forceX Pointer to the x force on the vertex that belongs to this
 *    thread.
 * \param[out] forceY Pointer to the y force on the vertex that belongs to this
 *    thread.
 * \param[in] deg Array holding the out degree values for each vertex.
 */
__device__ void fa2Repulsion(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg);

/*!
 * Updates the current force on each vertex with the current attraction.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] vxLocs Array that holds the x location of all vertices.
 * \param[in] vyLocs Array that holds the y location of all vertices.
 * \param[in] numedges The total number of edges in the graph.
 * \param[in] edgeSources Array holding the vertex ID for each edge source in
 *    order.
 * \param[in] edgeTargets Array holding the vertex ID for each edge target in
 *    order.
 * \param[out] forceX Pointer to the x force on the vertex that belongs to this
 *    thread.
 * \param[out] forceY Pointer to the y force on the vertex that belongs to this
 *    thread.
 */
__device__ void fa2Attraction(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, unsigned int numedges,
    unsigned int* edgeSources, unsigned int* edgeTargets, float* forceX,
    float* forceY);

/*!
 * Updates the swing value for each vertex in the graph.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] forceX The x force value on the vertex that belongs to this
 *    thread.
 * \param[in] forceY The y force value on the vertex that belongs to this
 *    thread.
 * \param[in] oldForceX Array holding the x forces of each vertex from the
 *    previous iteration.
 * \param[in] oldForceY Array holding the y forces of each vertex from the
 *    previous iteration.
 * \param[out] swg Array where the swing values for each vertex should be
 *    stored.
 */
__device__ void fa2UpdateSwing(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY, float* swg);

/*!
 * Updates the traction value for each vertex in the graph.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] forceX The x force value on the vertex that belongs to this
 *    thread.
 * \param[in] forceY The y force value on the vertex that belongs to this
 *    thread.
 * \param[in] oldForceX Array holding the x forces of each vertex from the
 *    previous iteration.
 * \param[in] oldForceY Array holding the y forces of each vertex from the
 *    previous iteration.
 * \param[out] swg Array where the traction values for each vertex should be
 *    stored.
 */
__device__ void fa2UpdateTract(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY, float* tra);

/*!
 * Updates the swing value for the graph itself.
 *
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] swg Array holding the swing values of each vertex in the graph.
 * \param[in] deg Array holding the out degree values of each vertex in the
 *    graph.
 * \param[out] gswing Pointer to where the graph swing value should be stored.
 */
__device__ void fa2UpdateSwingGraph(unsigned int numvertices, float* swg,
    int* deg, float* gswing);

/*!
 * Updates the traction value for the graph itself.
 *
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] tra Array holding the traction values of each vertex in the graph.
 * \param[in] deg Array holding the out degree values of each vertex in the
 *    graph.
 * \param[out] gtract Pointer to where the graph traction value should be stored.
 */
__device__ void fa2UpdateTractGraph(unsigned int numvertices, float* tra,
    int* deg, float* gtract);

/*!
 * Updates the speed value for the graph itself.
 *
 * \param[in] gswing The swing value of the graph.
 * \param[in] gtract The traction value of the graph.
 * \param[out] gspeed Pointer where the graph speed should be stored.
 */
__device__ void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed);

/*!
 * Updates the speed value for each vertex in the graph.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[out] speed Pointer to where the speed value of the vertex that belongs
 *    to this thread should be stored.
 * \param[in] forceX The x force on the vertex that belongs to this thread.
 * \param[in] forceY The y force on the vertex that belongs to this thread.
 * \param[in] gs The graph speed value.
 */
__device__ void fa2UpdateSpeed(unsigned int gid, unsigned int numvertices,
    float* speed, float* swg, float forceX, float forceY, float gs);

/*!
 * Copies the forces of this iteration to another array. Overwrites the values
 * in the destination array.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] forceX The x force on the vertex that belongs to this thread.
 * \param[in] forceY The y force on the vertex that belongs to this thread.
 * \param[out] oldForceX Array that will be used to store the x forces of this
 *    iteration.
 * \param[out] oldForceY Array that will be used to store the y forces of this
 *    iteration.
 */
__device__ void fa2SaveOldForces(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY);

/*!
 * Updates the vertex displacement array.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] speed The speed value for the vertex that belongs to this thread.
 * \param[in] forceX The x force value for the vertex that belongs to this
 *    thread.
 * \param[in] forceY The y force value for the vertex that belongs to this
 *    thread.
 * \param[out] dispX Pointer to where the x displacement for the vertex that
 *    belongs to this thread should be stored.
 * \param[out] dispY Pointer to where the y displacement for the vertex that
 *    belongs to this thread should be stored.
 */
__device__ void fa2UpdateDisplacement(unsigned int gid,
    unsigned int numvertices, float speed, float forceX, float forceY,
    float* dispX, float* dispY);

/*!
 * Updates the location of each vertex.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[out] vxLocs Array containing the x location of every vertex.
 * \param[out] vyLocs Array containing the y location of every vertex.
 * \param[in] xdisp The x displacement value for the vertex that belongs to this
 *    thread.
 * \param[in] ydisp The y displacement value for the vertex that belongs to this
 *    thread.
 */
__device__ void fa2UpdateLocation(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float xdisp, float ydisp);

/*!
 * Computes the out degree for each vertex.
 *
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] numedges The total number of edges in the graph.
 * \param[in] edgeSources Array holding the vertex index for each edge-source.
 * \param[out] deg Array holding the out degree for each vertex.
 */
__global__ void fa2ComputeDegrees(unsigned int numvertices,
    unsigned int numedges, unsigned int* edgeSources, unsigned int* deg);

/*!
 * CUDA Kernel that computes the graph swing and graph traction values.
 *
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[in] swg Array holding the swing value for each vertex in the graph.
 * \param[in] tra Array holding the traction value for each vertex in the
 *    graph.
 * \param[in] numNeighbours Array holding the out degree value for each vertex
 *    in the graph.
 * \param[out] graphSwing Pointer to where the graph swing value should be
 *    stored.
 * \param[out] graphTract Pointer to where the graph traction value should be
 *    stored.
 */
__global__ void fa2GraphSwingTract(unsigned int numvertices, float* swg,
    float* tra, unsigned int* numNeighbours, float* graphSwing,
    float* graphTract);

__device__ void fa2Gravity(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg)
{
  if (gid < numvertices)
  {
    float vx = vxLocs[gid];
    float vy = vyLocs[gid];
    float vlen = vectorGetLength(vx, vy);
    vectorInverse(&vx, &vy);
    vectorMultiply(&vx, &vy, K_G * (deg[gid] + 1) / vlen);
    vectorAdd(forceX, forceY, vx, vy);
  }
}

__device__ void fa2Repulsion(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg)
{
  if (gid < numvertices)
  {
    for (size_t j = 0; j < numvertices; j++)
    {
      if (gid == j)
        continue;
      float vx1 = vxLocs[gid];
      float vy1 = vyLocs[gid];
      float vx2 = vxLocs[j];
      float vy2 = vyLocs[j];

      vectorSubtract(&vx1, &vy1, vx2, vy2);
      float dist = vectorGetLength(vx1, vy1);

      if (dist > 0)
      {
        vectorNormalize(&vx1, &vy1);
        vectorMultiply(&vx1, &vy1,
            K_R * (((deg[gid] + 1) * (deg[j] + 1)) / dist));
        // vectorMultiply(&vx1, &vy1, 0.5);

        vectorAdd(forceX, forceY, vx1, vy1);
      }
    }
  }
}

__device__ void fa2Attraction(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, unsigned int numedges,
    unsigned int* edgeSources, unsigned int* edgeTargets, float* forceX,
    float* forceY)
{
  if (gid < numvertices)
  {
    float vx1 = vxLocs[gid];
    float vy1 = vyLocs[gid];
    // Complete scan on edge array.
    for (size_t i = 0; i < numedges; i++)
    {
      unsigned int source = edgeSources[i];
      unsigned int target = edgeTargets[i];
      // Edge source is this vertex.
      if (source == gid)
      {
        // Compute attraction force.
        float vx2 = vxLocs[target];
        float vy2 = vyLocs[target];

        vectorSubtract(&vx2, &vy2, vx1, vy1);
        // vectorMultiply(&vx2, &vy2, 0.5);
        vectorAdd(forceX, forceY, vx2, vy2);
      }
    }
  }
}

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
__device__ void fa2UpdateSwing(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY, float* swg)
{
  if (gid < numvertices)
  {
    float fx = oldForceX[gid];
    float fy = oldForceY[gid];
    vectorSubtract(&fx, &fy, forceX, forceY);
    float vlen = vectorGetLength(fx, fy);
    swg[gid] = vlen;
  }
}

// Updates the traction for each vertex, as described in the Force Atlas 2
// paper.
__device__ void fa2UpdateTract(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY, float* tra)
{
  if (gid < numvertices)
  {
    float fx = oldForceX[gid];
    float fy = oldForceY[gid];
    vectorAdd(&fx, &fy, forceX, forceY);
    float vlen = vectorGetLength(fx, fy);
    tra[gid] = vlen / 2;
  }
}

// Calculate the current swing of the graph.
__device__ void fa2UpdateSwingGraph(unsigned int numvertices, float* swg,
    unsigned int* deg, float* gswing)
{
  __shared__ float scratch[BLOCK_SIZE * 2];

  // Setup local data to perform reduction.
  unsigned int tx = threadIdx.x;
  unsigned int base = tx + (blockIdx.x * BLOCK_SIZE * 2);
  unsigned int stride = BLOCK_SIZE;

  if (base < numvertices)
  {
    scratch[tx] = (deg[base] + 1) * swg[base];
  }
  else
    scratch[tx] = 0;

  if (base + stride < numvertices)
  {
    scratch[tx + stride] = (deg[base + stride] + 1) * swg[base + stride];
  }
  else
    scratch[tx + stride] = 0;

  // Do block-local reduction.
  while (stride > 0)
  {
    __syncthreads();
    if (tx < stride)
    {
      scratch[tx] += scratch[tx + stride];
    }

    stride >>= 1;
  }

  // Do atomic add per block to obtain final value.
  __syncthreads();
  if (tx == 0)
    atomicAdd(gswing, scratch[tx]);
}

// Calculate the current traction of the graph.
__device__ void fa2UpdateTractGraph(unsigned int numvertices, float* tra,
    unsigned int* deg, float* gtract)
{
  __shared__ float scratch[BLOCK_SIZE * 2];

  // Setup local data to perform reduction.
  unsigned int tx = threadIdx.x;
  unsigned int base = tx + (blockIdx.x * BLOCK_SIZE * 2);
  unsigned int stride = BLOCK_SIZE;

  if (base < numvertices)
  {
    scratch[tx] = (deg[base] + 1) * tra[base];
  }
  else
    scratch[tx] = 0;

  if (base + stride < numvertices)
  {
    scratch[tx + stride] = (deg[base + stride] + 1) * tra[base + stride];
  }
  else
    scratch[tx + stride] = 0;

  // Do block-local reduction.
  while (stride > 0)
  {
    __syncthreads();
    if (tx < stride)
    {
      scratch[tx] += scratch[tx + stride];
    }

    stride >>= 1;
  }

  // Do atomic add per block to obtain final value.
  __syncthreads();
  if (tx == 0)
    atomicAdd(gtract, scratch[tx]);
}

__device__ void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed)
{
  float oldSpeed = *gspeed;

  if (gswing == 0)
  {
    gswing = FLOAT_EPSILON;
  }

  *gspeed = TAU * (gtract / gswing);

  if (oldSpeed > 0 && *gspeed > 1.5 * oldSpeed)
  {
    *gspeed = 1.5 * oldSpeed;
  }

}

__device__ void fa2UpdateSpeed(unsigned int gid, unsigned int numvertices,
    float* speed, float* swg, float forceX, float forceY, float gs)
{
  if (gid < numvertices)
  {
    float vSwg = swg[gid];
    if (vSwg <= 0)
      vSwg = EPSILON;
    float vForceLen = vectorGetLength(forceX, forceY);
    if (vForceLen <= 0)
      vForceLen = EPSILON;

    *speed = K_S * gs / (1 + (gs * sqrt(vSwg)));
  }
}

// Save current forces as the previous forces for the next tick.
__device__ void fa2SaveOldForces(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY)
{
  if (gid < numvertices)
  {
    oldForceX[gid] = forceX;
    oldForceY[gid] = forceY;
  }
}

__device__ void fa2UpdateDisplacement(unsigned int gid,
    unsigned int numvertices, float speed, float forceX, float forceY,
    float* dispX, float* dispY)
{
  if (gid < numvertices)
  {
    *dispX = forceX;
    *dispY = forceY;
    vectorMultiply(dispX, dispY, speed);
  }
}

__device__ void fa2UpdateLocation(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float xdisp, float ydisp)
{
  if (gid < numvertices)
  {
    vxLocs[gid] += xdisp;
    vyLocs[gid] += ydisp;
  }
}

__global__ void fa2ComputeDegrees(unsigned int numvertices,
    unsigned int numedges, unsigned int* edgeSources, unsigned int* deg)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  if (gid < numvertices)
  {
    unsigned int degree = 0;
    for (size_t i = 0; i < numedges; i++)
    {
      if (edgeSources[i] == gid)
      {
        degree++;
      }
    }
    deg[gid] = degree;
  }
}

__global__ void fa2GraphSwingTract(unsigned int numvertices, float* swg,
    float* tra, unsigned int* numNeighbours, float* graphSwing,
    float* graphTract)
{
  // Update swing of Graph.
  fa2UpdateSwingGraph(numvertices, swg, numNeighbours, graphSwing);

  // Update traction of Graph.
  fa2UpdateTractGraph(numvertices, tra, numNeighbours, graphTract);
}

__global__ void fa2kernel(float* vxLocs, float* vyLocs,
    unsigned int numvertices, unsigned int* edgeSources,
    unsigned int* edgeTargets, unsigned int numedges,
    unsigned int* numNeighbours, float* tra, float* swg, float* forceX,
    float* forceY, float* oldForceX, float* oldForceY, float* graphSwing,
    float* graphTract, float* oldGraphSpeed)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

  if (gid < numvertices)
  {
    forceX[gid] = 0;
    forceY[gid] = 0;

    // Gravity force
    fa2Gravity(gid, numvertices, vxLocs, vyLocs, &forceX[gid], &forceY[gid],
        numNeighbours);
    // Repulsion between vertices
    fa2Repulsion(gid, numvertices, vxLocs, vyLocs, &forceX[gid], &forceY[gid],
        numNeighbours);
    // Attraction on edges
    fa2Attraction(gid, numvertices, vxLocs, vyLocs, numedges, edgeSources,
        edgeTargets, &forceX[gid], &forceY[gid]);

    // Calculate speed of vertices.
    // Update swing of vertices.
    fa2UpdateSwing(gid, numvertices, forceX[gid], forceY[gid], oldForceX,
        oldForceY, swg);

    // Update traction of vertices.
    fa2UpdateTract(gid, numvertices, forceX[gid], forceY[gid], oldForceX,
        oldForceY, tra);
  }
}

__global__ void fa2MoveVertices(float* vxLocs, float* vyLocs,
    unsigned int numvertices, float* tra, float* swg, float* forceX,
    float* forceY, float* oldForceX, float* oldForceY, float* graphSwing,
    float* graphTract, float* graphSpeed)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

  if (gid < numvertices)
  {
    float speed = 0;
    float dispX = 0;
    float dispY = 0;

    // Update speed of Graph.
    fa2UpdateSpeedGraph(*graphSwing, *graphTract, graphSpeed);

    // Update speed of vertices.
    fa2UpdateSpeed(gid, numvertices, &speed, swg, forceX[gid], forceY[gid],
        *graphSpeed);

    // Update displacement of vertices.
    fa2UpdateDisplacement(gid, numvertices, speed, forceX[gid], forceY[gid],
        &dispX, &dispY);

    // Update vertex locations based on speed.
    fa2UpdateLocation(gid, numvertices, vxLocs, vyLocs, dispX, dispY);

    // Set current forces as old forces in vertex data.
    fa2SaveOldForces(gid, numvertices, forceX[gid], forceY[gid], oldForceX,
        oldForceY);
  }
}

void fa2PrepareGeneralMemory(ForceAtlas2Data* data, unsigned int numvertices)
{
  // Allocate data for vertices, edges, and fa2 data.
  cudaMalloc(&data->numNeighbours, numvertices * sizeof(int));
  cudaMalloc(&data->tra, numvertices * sizeof(float));
  cudaMalloc(&data->swg, numvertices * sizeof(float));
  cudaMalloc(&data->forceX, numvertices * sizeof(float));
  cudaMalloc(&data->forceY, numvertices * sizeof(float));
  cudaMalloc(&data->oldForceX, numvertices * sizeof(float));
  cudaMalloc(&data->oldForceY, numvertices * sizeof(float));
  cudaMalloc(&data->graphSwing, sizeof(float));
  cudaMalloc(&data->graphTract, sizeof(float));
  cudaMalloc(&data->graphSpeed, sizeof(float));

  cudaMemset(data->numNeighbours, 0, numvertices * sizeof(int));
  cudaMemset(data->tra, 0, numvertices * sizeof(float));
  cudaMemset(data->swg, 0, numvertices * sizeof(float));
  cudaMemset(data->forceX, 0, numvertices * sizeof(float));
  cudaMemset(data->forceY, 0, numvertices * sizeof(float));
  cudaMemset(data->oldForceX, 0, numvertices * sizeof(float));
  cudaMemset(data->oldForceY, 0, numvertices * sizeof(float));
  cudaMemset(data->graphSwing, 0, sizeof(float));
  cudaMemset(data->graphTract, 0, sizeof(float));
  cudaMemset(data->graphSpeed, 0, sizeof(float));

  /* These variables are not memsetted because data needs to be copied here */
  cudaMalloc(&data->vxLocs, numvertices * sizeof(float));
  cudaMalloc(&data->vyLocs, numvertices * sizeof(float));
}

void fa2PrepareEdgeMemory(ForceAtlas2Data* data, unsigned int numedges)
{
  cudaMalloc(&data->edgeSources, numedges * sizeof(unsigned int));
  cudaMalloc(&data->edgeTargets, numedges * sizeof(unsigned int));
}

void fa2PrepareMemory(ForceAtlas2Data* data, unsigned int numvertices,
    unsigned int numedges)
{
  fa2PrepareGeneralMemory(data, numvertices);
  fa2PrepareEdgeMemory(data, numedges);
}

void fa2CleanGeneralMemory(ForceAtlas2Data* data)
{
  cudaFree(data->numNeighbours);
  cudaFree(data->tra);
  cudaFree(data->swg);
  cudaFree(data->forceX);
  cudaFree(data->forceY);
  cudaFree(data->oldForceX);
  cudaFree(data->oldForceY);
  cudaFree(data->graphSwing);
  cudaFree(data->graphTract);
  cudaFree(data->graphSpeed);
  cudaFree(data->vxLocs);
  cudaFree(data->vyLocs);
}

void fa2CleanEdgeMemory(ForceAtlas2Data* data)
{
  cudaFree(data->edgeSources);
  cudaFree(data->edgeTargets);
}

void fa2CleanMemory(ForceAtlas2Data* data)
{
  fa2CleanGeneralMemory(data);
  fa2CleanEdgeMemory(data);
}

void fa2RunOnGraph(Graph* g, unsigned int iterations)
{
  CudaTimer timerMem1, timerMem2, timerIteration, timer;

  // Allocate data for vertices, edges, and fa2 data.
  ForceAtlas2Data data;
  fa2PrepareMemory(&data, g->vertices->numvertices, g->edges->numedges);

  startCudaTimer(&timerMem1);

  // Copy vertices and edges to device.
  cudaMemcpy((void*) data.vxLocs, g->vertices->vertexXLocs,
      g->vertices->numvertices * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*) data.vyLocs, g->vertices->vertexYLocs,
      g->vertices->numvertices * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*) data.edgeSources, g->edges->edgeSources,
      g->edges->numedges * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy((void*) data.edgeTargets, g->edges->edgeTargets,
      g->edges->numedges * sizeof(unsigned int), cudaMemcpyHostToDevice);

  stopCudaTimer(&timerMem1);

  unsigned int numblocks = ceil(g->vertices->numvertices / (float) BLOCK_SIZE);
  unsigned int numblocks_reduction = ceil(numblocks / 2.0);

  cudaGetLastError();

  // Compute vertex degrees using current edges.
  fa2ComputeDegrees<<<numblocks, BLOCK_SIZE>>>(g->vertices->numvertices,
      g->edges->numedges, data.edgeSources, data.numNeighbours);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess)
  {
    printf("Error calculating node degrees.\n%s\n", cudaGetErrorString(code));
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < iterations; i++)
  {
    // Run fa2 spring embedding kernel.
    startCudaTimer(&timerIteration);

    // Compute graph speed, vertex forces, speed and displacement.
    startCudaTimer(&timer);
    fa2kernel<<<numblocks, BLOCK_SIZE>>>(data.vxLocs, data.vyLocs,
        g->vertices->numvertices, data.edgeSources, data.edgeTargets,
        g->edges->numedges, data.numNeighbours, data.tra, data.swg, data.forceX,
        data.forceY, data.oldForceX, data.oldForceY, data.graphSwing,
        data.graphTract, data.graphSpeed);
    stopCudaTimer(&timer);
    printf("time: all forces and moving vertices.\n");
    printCudaTimer(&timer);
    resetCudaTimer(&timer);
    code = cudaGetLastError();
    if (code != cudaSuccess)
    {
      printf("Error in kernel 2.\n%s\n", cudaGetErrorString(code));
      exit(EXIT_FAILURE);
    }

    stopCudaTimer(&timerIteration);
    printf("time: iteration.\n");
    printCudaTimer(&timerIteration);
    resetCudaTimer(&timerIteration);

    cudaMemset(data.graphSwing, 0, sizeof(float));
    cudaMemset(data.graphTract, 0, sizeof(float));

    // Run reductions on vertex swing and traction.
    startCudaTimer(&timer);
    fa2GraphSwingTract<<<numblocks_reduction, BLOCK_SIZE>>>(
        g->vertices->numvertices, data.swg, data.tra, data.numNeighbours,
        data.graphSwing, data.graphTract);
    stopCudaTimer(&timer);
    printf("time: graph swing and traction.\n");
    printCudaTimer(&timer);
    resetCudaTimer(&timer);

    code = cudaGetLastError();
    if (code != cudaSuccess)
    {
      printf("Error in kernel 1.\n%s\n", cudaGetErrorString(code));
      exit(EXIT_FAILURE);
    }

    fa2MoveVertices<<<numblocks, BLOCK_SIZE>>>(data.vxLocs, data.vyLocs,
        g->vertices->numvertices, data.tra, data.swg, data.forceX, data.forceY,
        data.oldForceX, data.oldForceY, data.graphSwing, data.graphTract,
        data.graphSpeed);
  }

  startCudaTimer(&timerMem2);

  // Update graph with new vertex positions.
  cudaMemcpy((void*) g->vertices->vertexXLocs, data.vxLocs,
      g->vertices->numvertices * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy((void*) g->vertices->vertexYLocs, data.vyLocs,
      g->vertices->numvertices * sizeof(float), cudaMemcpyDeviceToHost);

  stopCudaTimer(&timerMem2);
  printf("time: copying data from host to device.\n");
  printCudaTimer(&timerMem1);
  printf("time: copying data from device to host.\n");
  printCudaTimer(&timerMem2);
  resetCudaTimer(&timerMem1);
  resetCudaTimer(&timerMem2);

  fa2CleanMemory(&data);
}

void fa2RunOnGraphInStream(Vertices* verticesIn, Edges** edgesIn, unsigned int numgraphs,
    unsigned int iterations, Vertices** verticesOut)
{
  CudaTimer timerMem1, timerMem2, timerIteration, timer;
  cudaStream_t* streams = createCudaStreams(numgraphs);

  ForceAtlas2Data data;
  fa2PrepareGeneralMemory(&data, verticesIn->numvertices);

  for (unsigned int timestep = 0; timestep < numgraphs; timestep++)
  {
    // Allocate data for vertices, edges, and fa2 data.
    ForceAtlas2Data edgeData;
    Edges* edges = edgesIn[timestep];
    cudaStream_t stream = streams[timestep];

    fa2PrepareEdgeMemory(&edgeData, edges->numedges);

    startCudaTimer(&timerMem1);

    // Copy vertices and edges to device.
    cudaMemcpyAsync((void*) data.vxLocs, verticesIn->vertexXLocs,
        verticesIn->numvertices * sizeof(float), cudaMemcpyHostToDevice,
        stream);
    cudaMemcpyAsync((void*) data.vyLocs, verticesIn->vertexYLocs,
        verticesIn->numvertices * sizeof(float), cudaMemcpyHostToDevice,
        stream);
    cudaMemcpyAsync((void*) edgeData.edgeSources, edges->edgeSources,
        edges->numedges * sizeof(unsigned int), cudaMemcpyHostToDevice,
        stream);
    cudaMemcpyAsync((void*) edgeData.edgeTargets, edges->edgeTargets,
        edges->numedges * sizeof(unsigned int), cudaMemcpyHostToDevice,
        stream);

    stopCudaTimer(&timerMem1);

    unsigned int numblocks = ceil(
        verticesIn->numvertices / (float) BLOCK_SIZE);
    unsigned int numblocks_reduction = ceil(numblocks / 2.0);

    cudaGetLastError();

    // Compute vertex degrees using current edges.
    fa2ComputeDegrees<<<numblocks, BLOCK_SIZE, 0, stream>>>(
        verticesIn->numvertices, edges->numedges, edgeData.edgeSources,
        data.numNeighbours);

    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
    {
      printf("Error calculating node degrees.\n%s\n", cudaGetErrorString(code));
      exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < iterations; i++)
    {
      // Run fa2 spring embedding kernel.
      startCudaTimer(&timerIteration);

      // Compute graph speed, vertex forces, speed and displacement.
      startCudaTimer(&timer);
      fa2kernel<<<numblocks, BLOCK_SIZE, 0, stream>>>(data.vxLocs, data.vyLocs,
          verticesIn->numvertices, edgeData.edgeSources, edgeData.edgeTargets,
          edges->numedges, data.numNeighbours, data.tra, data.swg,
          data.forceX, data.forceY, data.oldForceX, data.oldForceY,
          data.graphSwing, data.graphTract, data.graphSpeed);
      stopCudaTimer(&timer);
      printf("time: all forces and moving vertices.\n");
      printCudaTimer(&timer);
      resetCudaTimer(&timer);
      code = cudaGetLastError();
      if (code != cudaSuccess)
      {
        printf("Error in kernel 2.\n%s\n", cudaGetErrorString(code));
        exit(EXIT_FAILURE);
      }

      stopCudaTimer(&timerIteration);
      printf("time: iteration.\n");
      printCudaTimer(&timerIteration);
      resetCudaTimer(&timerIteration);

      cudaMemset(data.graphSwing, 0, sizeof(float));
      cudaMemset(data.graphTract, 0, sizeof(float));

      // Run reductions on vertex swing and traction.
      startCudaTimer(&timer);
      fa2GraphSwingTract<<<numblocks_reduction, BLOCK_SIZE, 0, stream>>>(
          verticesIn->numvertices, data.swg, data.tra, data.numNeighbours,
          data.graphSwing, data.graphTract);
      stopCudaTimer(&timer);
      printf("time: graph swing and traction.\n");
      printCudaTimer(&timer);
      resetCudaTimer(&timer);

      code = cudaGetLastError();
      if (code != cudaSuccess)
      {
        printf("Error in kernel 1.\n%s\n", cudaGetErrorString(code));
        exit(EXIT_FAILURE);
      }

      fa2MoveVertices<<<numblocks, BLOCK_SIZE, 0, stream>>>(data.vxLocs,
          data.vyLocs, verticesIn->numvertices, data.tra, data.swg,
          data.forceX, data.forceY, data.oldForceX, data.oldForceY,
          data.graphSwing, data.graphTract, data.graphSpeed);
    }

    startCudaTimer(&timerMem2);

    // Update graph with new vertex positions.
    cudaMemcpyAsync((void*) verticesOut[timestep]->vertexXLocs, data.vxLocs,
        verticesIn->numvertices * sizeof(float), cudaMemcpyDeviceToHost,
        stream);
    cudaMemcpyAsync((void*) verticesOut[timestep]->vertexYLocs, data.vyLocs,
        verticesIn->numvertices * sizeof(float), cudaMemcpyDeviceToHost,
        stream);

    stopCudaTimer(&timerMem2);
    printf("time: copying data from host to device.\n");
    printCudaTimer(&timerMem1);
    printf("time: copying data from device to host.\n");
    printCudaTimer(&timerMem2);
    resetCudaTimer(&timerMem1);
    resetCudaTimer(&timerMem2);

    fa2CleanEdgeMemory(&edgeData);
  }
  fa2CleanGeneralMemory(&data);
}

