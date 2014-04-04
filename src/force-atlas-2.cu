
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
    unsigned int* edgeTargets, float* forceX, float* forceY);

__device__ void fa2UpdateSwing(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY,
    float* swg);
__device__ void fa2UpdateTract(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY,
    float* tra);
__device__ void fa2UpdateSwingGraph(unsigned int gid, unsigned int numvertices,
    float* swg, int* deg, float* gswing);
__device__ void fa2UpdateTractGraph(unsigned int gid, unsigned int numvertices,
    float* tra, int* deg, float* gtract);
__device__ void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed);
__device__ void fa2UpdateSpeed(unsigned int gid, unsigned int numvertices,
    float* speed, float* swg, float forceX, float forceY, float gs);
__device__ void fa2SaveOldForces(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY);
__device__ void fa2UpdateDisplacement(unsigned int gid,
    unsigned int numvertices, float speed, float forceX, float forceY,
    float* dispX, float* dispY);
__device__ void fa2UpdateLocation(unsigned int gid, unsigned int numvertices,
    float* vxLocs, float* vyLocs, float xdisp, float ydisp);

__global__ void fa2ComputeDegrees(unsigned int numvertices,
    unsigned int numedges, unsigned int* edgeSources, unsigned int* deg);
__global__ void fa2GraphSwingTract(unsigned int numvertices,
    float* swg, float* tra, unsigned int* numNeighbours,
    float* graphSwing, float* graphTract);

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

      vectorAdd(forceX, forceY, vx1, vy1);
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
      unsigned int target = edgeSources[i];
      // Edge source is this vertex.
      if (source == gid)
      {
        // Compute attraction force.
        float vx2 = vxLocs[target];
        float vy2 = vyLocs[target];

        vectorSubtract(&vx2, &vy2, vx1, vy1);
        vectorMultiply(&vx2, &vy2, 0.5);
        vectorAdd(forceX, forceY, vx2, vy2);
      }
    }
  }
}

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
__device__ void fa2UpdateSwing(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY,
    float* swg)
{
  if (gid < numvertices)
  {
    float fx = forceX;
    float fy = forceY;
    vectorSubtract(&fx, &fy, oldForceX[gid], oldForceY[gid]);
    float vlen = vectorGetLength(fx, fy);
    swg[gid] = vlen;
  }
}

// Updates the traction for each vertex, as described in the Force Atlas 2
// paper.
__device__ void fa2UpdateTract(unsigned int gid, unsigned int numvertices,
    float forceX, float forceY, float* oldForceX, float* oldForceY,
    float* tra)
{
  if (gid < numvertices)
  {
    float fx = forceX;
    float fy = forceY;
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
  __syncthreads();
  if (tx == 0)
    atomicAdd(gswing, scratch[tx]);
}

// Calculate the current traction of the graph.
__device__ void fa2UpdateTractGraph(unsigned int gid, unsigned int numvertices,
    float* tra, unsigned int* deg, float* gtract)
{
  __shared__ float scratch[BLOCK_SIZE * 2];

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
  __syncthreads();
  if (tx == 0)
    atomicAdd(gtract, scratch[tx]);
}

__device__ void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed)
{
    float oldSpeed = *gspeed;
    *gspeed = gswing > 0 ? TAU * (gtract / gswing) : EPSILON;
    if (*gspeed <= 0)
      *gspeed = EPSILON;
    // Do not allow more then 50% speed increase.
    if (oldSpeed > FLOAT_EPSILON && *gspeed > 1.5 * oldSpeed)
      *gspeed = 1.5 * oldSpeed;
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
    *speed = fmin(*speed, (float)
        K_SMAX / vForceLen);
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

__global__ void fa2GraphSwingTract(unsigned int numvertices,
    float* swg, float* tra, unsigned int* numNeighbours,
    float* graphSwing, float* graphTract)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

  // Update swing of Graph.
  fa2UpdateSwingGraph(gid, numvertices, swg, numNeighbours, graphSwing);

  // Update traction of Graph.
  fa2UpdateTractGraph(gid, numvertices, tra, numNeighbours, graphTract);
}

__global__ void fa2kernel(
    float* vxLocs, float* vyLocs,
    unsigned int numvertices,
    unsigned int* edgeSources,
    unsigned int* edgeTargets,
    unsigned int numedges,
    unsigned int* numNeighbours,
    float* tra, float* swg, 
    float* oldForceX, float* oldForceY,
    float* graphSwing,
    float* graphTract,
    float* oldGraphSpeed)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  float graphSpeed = *oldGraphSpeed;

  float forceX = 0.0;
  float forceY = 0.0;

  float dispX = 0.0;
  float dispY = 0.0;

  float speed = 0.0;

  // Update speed of Graph.
  fa2UpdateSpeedGraph(*graphSwing, *graphTract, &graphSpeed);

  if (gid == 0)
  {
    *oldGraphSpeed = graphSpeed;
    *graphSwing = 0.0;
    *graphTract = 0.0;
  }

  // Gravity force
  fa2Gravity(gid, numvertices, vxLocs, vyLocs, &forceX, &forceY, numNeighbours);
  // Repulsion between vertices
  fa2Repulsion(gid, numvertices, vxLocs, vyLocs, &forceX, &forceY, numNeighbours);
  // Attraction on edges
  fa2Attraction(gid, numvertices, vxLocs, vyLocs, numedges, edgeSources,
      edgeTargets, &forceX, &forceY);

  // Calculate speed of vertices.
  // Update swing of vertices.
  fa2UpdateSwing(gid, numvertices, forceX, forceY, oldForceX, oldForceY, swg);

  // Update traction of vertices.
  fa2UpdateTract(gid, numvertices, forceX, forceY, oldForceX, oldForceY, tra);

  // Update speed of vertices.
  fa2UpdateSpeed(gid, numvertices, &speed, swg, forceX, forceY, graphSpeed);

  // Update displacement of vertices.
  fa2UpdateDisplacement(gid, numvertices, speed, forceX, forceY, &dispX, &dispY);

  // Set current forces as old forces in vertex data.
  fa2SaveOldForces(gid, numvertices, forceX, forceY, oldForceX, oldForceY);

  // Update vertex locations based on speed.
  fa2UpdateLocation(gid, numvertices, vxLocs, vyLocs, dispX, dispY);
}

void fa2RunOnGraph(Graph* g, unsigned int iterations)
{
  // Make variables for vertices, edges and fa2 data.
  unsigned int* numNeighbours = NULL;
  float* tra = NULL;
  float* swg = NULL;
  float* oldForceX = NULL;
  float* oldForceY = NULL;
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
  cudaMalloc(&oldForceX, g->numvertices * sizeof(float));
  cudaMalloc(&oldForceY, g->numvertices * sizeof(float));
  cudaMalloc(&graphSwing, sizeof(float));
  cudaMalloc(&graphTract, sizeof(float));
  cudaMalloc(&graphSpeed, sizeof(float));

  cudaMemset(numNeighbours, 0, g->numvertices * sizeof(int));
  cudaMemset(tra, 0, g->numvertices * sizeof(float));
  cudaMemset(swg, 0, g->numvertices * sizeof(float));
  cudaMemset(oldForceX, 0, g->numvertices * sizeof(float));
  cudaMemset(oldForceY, 0, g->numvertices * sizeof(float));
  cudaMemset(graphSwing, 0, sizeof(float));
  cudaMemset(graphTract, 0, sizeof(float));
  cudaMemset(graphSpeed, 0, sizeof(float));

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
  unsigned int numblocks_reduction = ceil(numblocks / 2.0);

  // Compute vertex degrees using current edges.
  fa2ComputeDegrees<<<numblocks, BLOCK_SIZE>>>(g->numvertices, g->numedges,
      edgeSources, numNeighbours);

  for (size_t i = 0; i < iterations; i++)
  {
    // Run fa2 spring embedding kernel.

    // Run reductions on vertex swing and traction.
    fa2GraphSwingTract<<<numblocks_reduction, BLOCK_SIZE>>>(
        g->numvertices,
        swg, tra, numNeighbours,
        graphSwing, graphTract);

    // Compute graph speed, vertex forces, speed and displacement.
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
        oldForceX,
        oldForceY,
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

  cudaFree(numNeighbours);
  cudaFree(tra);
  cudaFree(swg);
  cudaFree(oldForceX);
  cudaFree(oldForceY);
  cudaFree(graphSwing);
  cudaFree(graphTract);
  cudaFree(graphSpeed);
}

