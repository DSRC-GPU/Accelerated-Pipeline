
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "force-atlas-2.h"
#include "math.h"
#include "vector.h"

#define K_R 1.0
#define K_S 0.1
#define K_G 5
#define K_SMAX 10.0
#define TAU 1.0
#define EPSILON 0.1
#define FLOAT_EPSILON 0.0000001

#define BLOCK_SIZE 64

// Gravity force
__device__ void fa2Gravity(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg);
// Repulsion between vertices
__device__ void fa2Repulsion(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg);
// Attraction on edges
__device__ void fa2Attraction(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, int numedges, unsigned int* edgeSources,
    unsigned int* edgeTargets, float* forceX, float* forceY, unsigned int* deg);

__device__ void fa2UpdateSwing(unsigned int gid, unsigned int numvertices,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY,
    float* swg);
__device__ void fa2UpdateTract(unsigned int gid, unsigned int numvertices,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY,
    float* tra);
__device__ void fa2UpdateSwingGraph(unsigned int gid, unsigned int numvertices,
    float* swg, int* deg, float* gswing);
__device__ void fa2UpdateTractGraph(unsigned int gid, unsigned int numvertices,
    float* tra, int* deg, float* gtract);
__device__ void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed);
__device__ void fa2UpdateSpeed(unsigned int gid, unsigned int numvertices,
    float* speed, float* swg, float* forceX, float* forceY, float gs);
__device__ void fa2SaveOldForces(unsigned int gid, unsigned int numvertices,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY);
__device__ void fa2UpdateDisplacement(unsigned int gid,
    unsigned int numvertices, float* speed, float* forceX, float* forceY,
    float* dispX, float* dispY);
__device__ void fa2UpdateLocation(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* xdisp, float* ydisp);
__device__ void fa2ResetForces(unsigned int gid, unsigned int numvertices,
    float* forceX, float* forceY);

__global__ void fa2ComputeDegrees(unsigned int numvertices,
    unsigned int numedges, unsigned int* edgeSources, unsigned int* deg);

__device__ void fa2Gravity(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg)
{
  if (gid < numvertices)
  {
    float vx = vxLocs[gid];
    float vy = vyLocs[gid];
    vectorNormalize(&vx, &vy);
    vectorInverse(&vx, &vy);
    vectorMultiply(&vx, &vy, K_G * (deg[gid] + 1));
    vectorAdd(&forceX[gid], &forceY[gid], vx, vy);
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
      if (gid == j) continue;
      float vx1 = vxLocs[gid];
      float vy1 = vyLocs[gid];
      float vx2 = vxLocs[j];
      float vy2 = vyLocs[j];

      vectorSubtract(&vx1, &vy1, vx2, vy2);
      float dist = vectorGetLength(vx1, vy1);
      if (dist < FLOAT_EPSILON)
        dist = EPSILON;

      vectorNormalize(&vx1, &vy1);
      vectorMultiply(&vx1, &vy1, K_R * (((deg[gid] + 1) * (deg[j] + 1))
            / dist));
      vectorMultiply(&vx1, &vy1, 0.5);

      vectorAdd(&forceX[gid], &forceY[gid], vx1, vy1);
    }
  }
}

__device__ void fa2Attraction(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, unsigned int numedges,
    unsigned int* edgeSources, unsigned int* edgeTargets, float* forceX,
    float* forceY, unsigned int* deg)
{
  if (gid < numvertices)
  {
    float vx1 = vxLocs[gid];
    float vy1 = vyLocs[gid];
    // Complete scan on edge array.
    for (size_t i = 0; i < numedges; i++)
    {
      unsigned int source = edgeSources[i];
      unsigned int target = edgeSources[i];
      // Edge source is this vertex.
      if (source == gid)
      {
        // Increase the degree of this node by one.
        deg[gid]++;

        // Compute attraction force.
        float vx2 = vxLocs[target];
        float vy2 = vyLocs[target];

        vectorSubtract(&vx2, &vy2, vx1, vy1);
        vectorMultiply(&vx2, &vy2, 0.5);
        vectorAdd(&forceX[gid], &forceY[gid], vx2, vy2);
      }
    }
  }
}

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
__device__ void fa2UpdateSwing(unsigned int gid, unsigned int numvertices,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY,
    float* swg)
{
  if (gid < numvertices)
  {
    float fx = forceX[gid];
    float fy = forceY[gid];
    vectorSubtract(&fx, &fy, oldForceX[gid], oldForceY[gid]);
    float vlen = vectorGetLength(fx, fy);
    swg[gid] = vlen;
  }
}

// Updates the traction for each vertex, as described in the Force Atlas 2
// paper.
__device__ void fa2UpdateTract(unsigned int gid, unsigned int numvertices,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY,
    float* tra)
{
  if (gid < numvertices)
  {
    float fx = forceX[gid];
    float fy = forceY[gid];
    vectorAdd(&fx, &fy, oldForceX[gid], oldForceY[gid]);
    float vlen = vectorGetLength(fx, fy);
    tra[gid] = vlen / 2;
  }
}

// Calculate the current swing of the graph.
__device__ void fa2UpdateSwingGraph(unsigned int gid, unsigned int numvertices,
    float* swg, unsigned int* deg, float* gswing)
{
  __shared__ float scratch[BLOCK_SIZE * 2];

  // Initialize output to 0.
  if (gid == 0)
    *gswing = 0;

  // Setup local data to perform reduction.
  unsigned int tx = threadIdx.x;
  unsigned int base = tx + (blockIdx.x * BLOCK_SIZE * 2);
  unsigned int stride = BLOCK_SIZE;

  if (base < numvertices)
    scratch[tx] = (deg[base] + 1) * swg[base];
  else
    scratch[tx] = 0;

  if (base + stride < numvertices)
    scratch[tx + stride] = (deg[base + stride] + 1) * swg[base + stride];
  else
    scratch[tx + stride] = 0;

  // Do block-local reduction.
  while (stride > 0)
  {
    __syncthreads();
    if (tx < stride)
    {
      scratch[base] += scratch[base + stride];
    }

    stride >>= 1;
  }

  // Do atomic add per block to obtain final value.
  if (tx == 0)
    atomicAdd(gswing, scratch[tx]);
}

// Calculate the current traction of the graph.
__device__ void fa2UpdateTractGraph(unsigned int gid, unsigned int numvertices,
    float* tra, unsigned int* deg, float* gtract)
{
  __shared__ float scratch[BLOCK_SIZE * 2];

  // Initialize output to 0.
  if (gid == 0)
    *gtract = 0;

  // Setup local data to perform reduction.
  unsigned int tx = threadIdx.x;
  unsigned int base = tx + (blockIdx.x * BLOCK_SIZE * 2);
  unsigned int stride = BLOCK_SIZE;

  if (base < numvertices)
    scratch[tx] = (deg[base] + 1) * tra[base];
  else
    scratch[tx] = 0;

  if (base + stride < numvertices)
    scratch[tx + stride] = (deg[base + stride] + 1) * tra[base + stride];
  else
    scratch[tx + stride] = 0;

  // Do block-local reduction.
  while (stride > 0)
  {
    __syncthreads();
    if (tx < stride)
    {
      scratch[base] += scratch[base + stride];
    }

    stride >>= 1;
  }

  // Do atomic add per block to obtain final value.
  if (tx == 0)
    atomicAdd(gtract, scratch[tx]);
}

__device__ void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed)
{
  // This code is executed by one thread.
  if (threadIdx.x + blockIdx.x == 0)
  {
    float oldSpeed = *gspeed;
    *gspeed = gswing > 0 ? TAU * (gtract / gswing) : EPSILON;
    if (*gspeed <= 0)
      *gspeed = EPSILON;
    // Do not allow more then 50% speed increase.
    if (oldSpeed > FLOAT_EPSILON && *gspeed > 1.5 * oldSpeed)
      *gspeed = 1.5 * oldSpeed;
  }
}

__device__ void fa2UpdateSpeed(unsigned int gid, unsigned int numvertices,
    float* speed, float* swg, float* forceX, float* forceY, float gs)
{
  if (gid < numvertices)
  {
    float vSwg = swg[gid];
    if (vSwg <= 0)
      vSwg = EPSILON;
    float vForceLen = vectorGetLength(forceX[gid], forceY[gid]);
    if (vForceLen <= 0)
      vForceLen = EPSILON;

    speed[gid] = K_S * gs / (1 + (gs * sqrt(vSwg)));
    speed[gid] = fmin(speed[gid], (float)
        K_SMAX / vForceLen);
  }
}

// Save current forces as the previous forces for the next tick.
__device__ void fa2SaveOldForces(unsigned int gid, unsigned int numvertices,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY)
{
  if (gid < numvertices)
  {
    oldForceX[gid] = forceX[gid];
    oldForceY[gid] = forceY[gid];
  }
}

__device__ void fa2UpdateDisplacement(unsigned int gid,
    unsigned int numvertices, float* speed, float* forceX, float* forceY,
    float* dispX, float* dispY)
{
  if (gid < numvertices)
  {
    dispX[gid] = forceX[gid];
    dispY[gid] = forceY[gid];
    vectorMultiply(&dispX[gid], &dispY[gid], speed[gid]);
  }
}

__device__ void fa2UpdateLocation(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* xdisp, float* ydisp)
{
  if (gid < numvertices)
  {
    vxLocs[gid] += xdisp[gid];
    vyLocs[gid] += ydisp[gid];
  }
}

__device__ void fa2ResetForces(unsigned int gid, unsigned int numvertices,
    float* forceX, float* forceY)
{
  if (gid < numvertices)
  {
    forceX[gid] = 0;
    forceY[gid] = 0;
  }
}

__global__ void fa2ComputeDegrees(unsigned int numvertices,
    unsigned int numedges, unsigned int* edgeSources, unsigned int* deg)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  if (gid < numvertices)
  {
    deg[gid] = 0;
    for (size_t i = 0; i < numedges; i++)
    {
      if (edgeSources[i] == gid)
      {
        deg[gid]++;
      }
    }
  }
}

__global__ void fa2kernel(
    float* vxLocs, float* vyLocs,
    unsigned int numvertices,
    unsigned int* edgeSources,
    unsigned int* edgeTargets,
    unsigned int numedges,
    unsigned int* numNeighbours,
    float* tra, float* swg, float* speed,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY,
    float* dispX, float* dispY,
    float* graphSwing,
    float* graphTract,
    float* graphSpeed)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

  // Gravity force
  fa2Gravity(gid, numvertices, vxLocs, vyLocs, forceX, forceY, numNeighbours);
  // Repulsion between vertices
  fa2Repulsion(gid, numvertices, vxLocs, vyLocs, forceX, forceY, numNeighbours);
  // Attraction on edges
  fa2Attraction(gid, numvertices, vxLocs, vyLocs, numedges, edgeSources,
      edgeTargets, forceX, forceY, numNeighbours);

  // Calculate speed of vertices.
  // Update swing of vertices.
  fa2UpdateSwing(gid, numvertices, forceX, forceY, oldForceX, oldForceY, swg);

  // Update traction of vertices.
  fa2UpdateTract(gid, numvertices, forceX, forceY, oldForceX, oldForceY, tra);

  // Update swing of Graph.
  fa2UpdateSwingGraph(gid, numvertices, swg, numNeighbours, graphSwing);

  // Update traction of Graph.
  fa2UpdateTractGraph(gid, numvertices, tra, numNeighbours, graphTract);

  // Update speed of Graph.
  fa2UpdateSpeedGraph(*graphSwing, *graphTract, graphSpeed);

  // Update speed of vertices.
  fa2UpdateSpeed(gid, numvertices, speed, swg, forceX, forceY, *graphSpeed);

  // Update displacement of vertices.
  fa2UpdateDisplacement(gid, numvertices, speed, forceX, forceY, dispX, dispY);

  // Set current forces as old forces in vertex data.
  fa2SaveOldForces(gid, numvertices, forceX, forceY, oldForceX, oldForceY);

  // Update vertex locations based on speed.
  fa2UpdateLocation(gid, numvertices, vxLocs, vyLocs, dispX, dispY);

  // set forces to 0.
  fa2ResetForces(gid, numvertices, forceX, forceY);
}

void fa2RunOnGraph(Graph* g, unsigned int iterations)
{
  // Make variables for vertices, edges and fa2 data.
  unsigned int* numNeighbours = NULL;
  float* tra = NULL;
  float* swg = NULL;
  float* speed = NULL;
  float* forceX = NULL;
  float* forceY = NULL;
  float* oldForceX = NULL;
  float* oldForceY = NULL;
  float* dispX = NULL;
  float* dispY = NULL;
  float* graphSwing = NULL;
  float* graphTract = NULL;
  float* graphSpeed = NULL;

  float* vxLocs = NULL;
  float* vyLocs = NULL;
  unsigned int* edgeSources = NULL;
  unsigned int* edgeTargets = NULL;

  // Allocate data for vertices, edges, and fa2 data.
  cudaMalloc(&numNeighbours, g->numvertices * sizeof(int));
  cudaMalloc(&tra, g->numvertices * sizeof(float));
  cudaMalloc(&swg, g->numvertices * sizeof(float));
  cudaMalloc(&speed, g->numvertices * sizeof(float));
  cudaMalloc(&forceX, g->numvertices * sizeof(float));
  cudaMalloc(&forceY, g->numvertices * sizeof(float));
  cudaMalloc(&oldForceX, g->numvertices * sizeof(float));
  cudaMalloc(&oldForceY, g->numvertices * sizeof(float));
  cudaMalloc(&dispX, g->numvertices * sizeof(float));
  cudaMalloc(&dispY, g->numvertices * sizeof(float));
  cudaMalloc(&graphSwing, sizeof(float));
  cudaMalloc(&graphTract, sizeof(float));
  cudaMalloc(&graphSpeed, sizeof(float));

  cudaMalloc(&vxLocs, g->numvertices * sizeof(float));
  cudaMalloc(&vyLocs, g->numvertices * sizeof(float));
  cudaMalloc(&edgeSources, g->numedges * sizeof(unsigned int));
  cudaMalloc(&edgeTargets, g->numedges * sizeof(unsigned int));

  // Copy vertices and edges to device.
  cudaMemcpy((void*) vxLocs, g->vertexXLocs, g->numvertices * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy((void*) vyLocs, g->vertexYLocs, g->numvertices * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy((void*) edgeSources, g->edgeSources,
      g->numedges * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy((void*) edgeTargets, g->edgeTargets,
      g->numedges * sizeof(unsigned int), cudaMemcpyHostToDevice);

  unsigned int numblocks = ceil(g->numvertices / (float) BLOCK_SIZE);

  // Compute vertex degrees using current edges.
  fa2ComputeDegrees<<<numblocks, BLOCK_SIZE>>>(g->numvertices, g->numedges,
      edgeSources, numNeighbours);

  for (size_t i = 0; i < iterations; i++)
  {
    // Run fa2 spring embedding kernel.
    fa2kernel<<<numblocks, BLOCK_SIZE>>>(
        vxLocs,
        vyLocs,
        g->numvertices,
        edgeSources,
        edgeTargets,
        g->numedges,
        numNeighbours,
        tra,
        swg,
        speed,
        forceX,
        forceY,
        oldForceX,
        oldForceY,
        dispX,
        dispY,
        graphSwing,
        graphTract,
        graphSpeed);
  }

  // Update graph with new vertex positions.
  cudaMemcpy((void*) g->vertexXLocs, vxLocs, g->numvertices * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaMemcpy((void*) g->vertexYLocs, vyLocs, g->numvertices * sizeof(float),
      cudaMemcpyDeviceToHost);
  cudaMemcpy((void*) g->edgeSources, edgeSources,
      g->numedges * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void*) g->edgeTargets, edgeTargets,
      g->numedges * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

