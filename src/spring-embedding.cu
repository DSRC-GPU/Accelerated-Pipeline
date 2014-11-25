/*!
 * \file force-atlas-2.cu
 * A parallel implementation of the Force Atlas 2 spring embedding algorithm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include "spring-embedding.h"
#include "math.h"
#include "timer.h"
#include "cuda-stream.h"
#include "vector.h"
#include "util.h"

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
__global__ void fa2Gravity(unsigned int numvertices,
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
__global__ void fa2Repulsion(unsigned int numvertices,
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
 * \param[in] edgeTargets Array holding the edge targets for all edges.
 * \param[in] maxedges The maximum number of edges per vertex.
 * \param[out] forceX Pointer to the x force on the vertex that belongs to this
 *    thread.
 * \param[out] forceY Pointer to the y force on the vertex that belongs to this
 *    thread.
 */
__global__ void fa2Attraction(unsigned int numvertices,
    float* vxLocs, float* vyLocs, unsigned int* numedges,
    unsigned int* edgeTargets, unsigned int maxedges, float* forceX,
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
__global__ void fa2UpdateSwing(unsigned int numvertices,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY, float* swg);

/*!
 * Updates the speed value for each vertex in the graph.
 *
 * \param[in] gid The global ID of this thread.
 * \param[in] numvertices The total number of vertices in the graph.
 * \param[out] speed Pointer to where the speed value of the vertex that belongs
 *    to this thread should be stored.
 * \param[in] swg An array holding the swing values for each vertex.
 * \param[in] forceX The x force on the vertex that belongs to this thread.
 * \param[in] forceY The y force on the vertex that belongs to this thread.
 * \param[in] gs The graph speed value.
 */
__global__ void fa2UpdateSpeed(unsigned int numvertices,
    float* speed, float* swg, float* forceX, float* forceY, float* gs);

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
__global__ void fa2GraphSwingTract(unsigned int numvertices,
    float** gGlobalArrays, unsigned int* numNeighbours, float* gGlobalVars);

__global__ void fa2Gravity(unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
#ifdef YIFAN_HU
#else
  if (gid < numvertices)
  {
    float vx = vxLocs[gid];
    float vy = vyLocs[gid];
    float vlen = sqrt(vx * vx + vy * vy);
    assert(vlen != 0);
    float factor = K_G * (deg[gid] + 1) / vlen;
    vx *= -factor;
    vy *= -factor;
    if (gid == 0)
      DEBUG_PRINT("g:%f\n", vx);
    vectorAdd(forceX, forceY, vx, vy);
    forceX[gid] += vx;
    forceY[gid] += vy;
  }
#endif
}

__global__ void fa2Repulsion(unsigned int numvertices,
    float* vxLocs, float* vyLocs, float* forceX, float* forceY,
    unsigned int* deg)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  // Local accumulators that will be written to global at end of computation.
  float tempVectorX = 0;
  float tempVectorY = 0;

  // Local copies of the location and outdegree of the vertex belonging to
  // this thread.
  float vx1 = vxLocs[gid];
  float vy1 = vyLocs[gid];
  float tDeg = deg[gid];

  // Allocating shared space for locations and outdegrees of other vertices.
  __shared__ float vxs[BLOCK_SIZE];
  __shared__ float vys[BLOCK_SIZE];
  __shared__ float sDeg[BLOCK_SIZE];

  // Tiled iteration. Chuck input in tiles.
  unsigned int numTiles = ceil((double) numvertices / BLOCK_SIZE);
  for (size_t j = 0; j < numTiles; j++)
  {
    // Calculate global index to use when loading element in shared mem.
    unsigned int index = (gid + j * BLOCK_SIZE) % (numTiles * BLOCK_SIZE);
    // Load location and outdegree into shared mem.
    unsigned int tid = threadIdx.x;
    if (index < numvertices)
    {
      vxs[tid] = vxLocs[index];
      vys[tid] = vyLocs[index];
      sDeg[tid] = deg[index];
    } else {
      vxs[tid] = 0;
      vys[tid] = 0;
      // This causes the enumerator later on to become 0, which in turn causes
      // the repulsion force to become 0.
      sDeg[tid] = -1;
    }
    // Sync to make sure shared mem has been filled.
    __syncthreads();

    // For all elements in the shared mem, compute repulsion.
    for (size_t i = 0; i < BLOCK_SIZE; i++)
    {
      // Create copy of the location of this vertex. Is overwritten when
      // computing the distance to vertex from shared mem.
      float xdist = vx1;
      float ydist = vy1;
      // Compute distance to other vertex and overwrite.
      xdist -= vxs[i];
      ydist -= vys[i];
      // Calculate euclidian distance to other vertex.
      float dist = sqrt(xdist * xdist + ydist * ydist);

      // Check because we are using dist as denominator.
      if (dist > 0)
      {
        // Shrink vertex to have length of 1.
        xdist /= dist;
        ydist /= dist;
        // Multiply by factor as specified in fa2 paper to compute repulsion
        // between vertices.
        xdist *= K_R * (((tDeg + 1) * (sDeg[i] + 1)) / dist);
        ydist *= K_R * (((tDeg + 1) * (sDeg[i] + 1)) / dist);
        // Add this repulsion to local variables.
        tempVectorX += xdist;
        tempVectorY += ydist;
      }
    }

    // Prevent threads from loading new values in shared mem while others are
    // still computing.
    __syncthreads();
  }
  // Add accumulated repulsion value to global mem variable.
  *forceX += tempVectorX;
  *forceY += tempVectorY;
}

__global__ void fa2Attraction(unsigned int numvertices,
    float* vxLocs, float* vyLocs, unsigned int* numedges,
    unsigned int* edgeTargets, unsigned int maxedges, float* forceX,
    float* forceY)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  if (gid < numvertices)
  {
    float vx1 = vxLocs[gid];
    float vy1 = vyLocs[gid];
    // Each thread goes through its array of edges.
    for (size_t i = 0; i < numedges[gid]; i++)
    {
      unsigned int index = gid + (numvertices * i);
      assert(index < numvertices + (numvertices * maxedges));
      unsigned int target = edgeTargets[index];
      assert(target < numvertices);
      // Compute attraction force.
      float vx2 = vxLocs[target];
      float vy2 = vyLocs[target];

      // v2 <- v2 - v1
      vectorSubtract(&vx2, &vy2, vx1, vy1);
#ifdef YIFAN_HU
      float dist = sqrt(vx2 * vx2 + vy2 * vy2);
      vx2 *= dist / YIFAN_HU_K;
      vy2 *= dist / YIFAN_HU_K;
#endif
      vectorAdd(forceX, forceY, vx2, vy2);
      if (gid == 0)
        DEBUG_PRINT("a:%f\t%u\n", vx2, target);
    }
  }
}

// Updates the swing for each vertex, as described in the Force Atlas 2 paper.
__global__ void fa2UpdateSwing(unsigned int numvertices,
    float* dForcesX, float* dForcesY, float* oldForceX, float* oldForceY, float* swg,
    float* tra)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  if (gid < numvertices)
  {
    float originalOldfx = oldForceX[gid];
    float originalOldfy = oldForceY[gid];
    float forceX = dForcesX[gid];
    float forceY = dForcesY[gid];

    float fx = originalOldfx - forceX;
    float fy = originalOldfy - forceY;
    swg[gid] = sqrt(fx * fx + fy * fy);

    fx = originalOldfx + forceX;
    fy = originalOldfy + forceY;
    tra[gid] = sqrt(fx * fx + fy * fy) / 2;
  }
}

/*!
 * Updates the swing value for the graph itself.
 *
 * \param[in] numElements The total number of vertices in the graph.
 * \param[in] gMultiplyArray Array holding the swing values of each vertex in the graph.
 * \param[in] gAccumulateArray Array holding the out gAccumulateArrayree values of each vertex in the
 *    graph.
 * \param[out] out Pointer to where the graph swing value should be stored.
 */
__device__ void arrayReduction(unsigned int numElements,
    float* gMultiplyArray, int incrementValue, unsigned int* gAccumulateArray,
    float* out)
{
  __shared__ float scratch[BLOCK_SIZE * 2];

  // Setup local data to perform reduction.
  unsigned int tx = threadIdx.x;
  unsigned int base = tx + (blockIdx.x * BLOCK_SIZE * 2);
  unsigned int stride = BLOCK_SIZE;

  if (base < numElements)
  {
    scratch[tx] = (gAccumulateArray[base] + incrementValue)
        * gMultiplyArray[base];
  }
  else
    scratch[tx] = 0;

  if (base + stride < numElements)
  {
    scratch[tx + stride] = (gAccumulateArray[base + stride] + incrementValue)
        * gMultiplyArray[base + stride];
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
    atomicAdd(out, scratch[tx]);
}

__device__ void arrayReductionFloat(unsigned int numElements,
    float* gMultiplyArray, int incrementValue, float* gAccumulateArray,
    float* out)
{
  __shared__ float scratch[BLOCK_SIZE * 2];

  // Setup local data to perform reduction.
  unsigned int tx = threadIdx.x;
  unsigned int base = tx + (blockIdx.x * BLOCK_SIZE * 2);
  unsigned int stride = BLOCK_SIZE;

  if (base < numElements)
  {
    scratch[tx] = (gAccumulateArray[base] + incrementValue)
        * gMultiplyArray[base];
  }
  else
    scratch[tx] = 0;

  if (base + stride < numElements)
  {
    scratch[tx + stride] = (gAccumulateArray[base + stride] + incrementValue)
        * gMultiplyArray[base + stride];
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
    atomicAdd(out, scratch[tx]);
}

#ifdef YIFAN_HU
#else
void fa2UpdateSpeedGraph(float gswing, float gtract, float* gspeed)
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

__global__ void fa2UpdateSpeed(unsigned int numvertices,
    float* speed, float* swg, float* dForceX, float* dForceY, float* globalVars)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  if (gid < numvertices)
  {
    float gs = globalVars[2];
    float vSwg = swg[gid];
    float forceX = dForceX[gid];
    float forceY = dForceY[gid];
    if (vSwg <= 0)
      vSwg = EPSILON;
    float vForceLen = sqrt(forceX * forceX + forceY * forceY);
    if (vForceLen <= 0)
      vForceLen = EPSILON;

    float vSpeed = K_S * gs / (1 + (gs * sqrt(vSwg)));
    float upperBound = K_SMAX / vForceLen;
    if (vSpeed > upperBound)
    {
      vSpeed = upperBound;
    }
    speed[gid] = vSpeed;
  }
}
#endif

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

__global__ void fa2GraphSwingTract(unsigned int numvertices,
    float** gGlobalArrays, unsigned int* numNeighbours, float* gGlobalVars)
{
#ifdef YIFAN_HU
  // Compute energy.
  // foreach vertex:
  //  energy += ||f||**2 (-> sqrt(a**2 + b**2)**2 -> a**2 + b**2)
  arrayReductionFloat(numvertices, gGlobalArrays[0], 0, gGlobalArrays[0],
      &gGlobalVars[0]);
  arrayReductionFloat(numvertices, gGlobalArrays[1], 0, gGlobalArrays[1],
      &gGlobalVars[0]);
#else // Force Atlas 2
  // Update swing of Graph.
  arrayReduction(numvertices, gGlobalArrays[0], 1, numNeighbours, &gGlobalVars[0]);

  // Update traction of Graph.
  arrayReduction(numvertices, gGlobalArrays[1], 1, numNeighbours, &gGlobalVars[1]);
#endif
}

__global__ void resetForcesKernel(unsigned int numvertices, float* forceX, float* forceY)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
  if (gid < numvertices)
  {
    forceX[gid] = 0;
    forceY[gid] = 0;
  }
}

/*!
 * Moves the vertices to their new location after force computation is complete.
 *
 * \param[in,out] vxLocs The x-locations of the vertices. This will be updated with new positions.
 * \param[in,out] vyLocs The y-locations of the vertices. This will be updated with new positions.
 * \param[in] numvertices The total number of vertices.
 * \param[in] tra An array holding the traction values for each vertex.
 * \param[in] swg An array holding the swing values for each vertex.
 * \param[in] forceX An array holding the x forces on each vertex.
 * \param[in] forceY An array holding the y forces on each vertex.
 * \param[in,out] oldForceX An array holding the x force on each vertex of the previous iteration.
 * These values will be overwritten with the current forces.
 * \param[in,out] oldForceY An array holding the y force on each vertex of the previous iteration.
 * These values will be overwritten with the current forces.
 * \param[in] graphSwing the swing value of the graph.
 * \param[in] graphTract the traction value of the graph.
 * \param[in] graphSpeed the speed value of the graph.
 */
__global__ void fa2MoveVertices(float* vxLocs, float* vyLocs,
    unsigned int numvertices, float* gGlobalVars, float* tra, float* swg,
    float* forceX, float* forceY, float* oldForceX, float* oldForceY,
    float* speed)
{
  unsigned int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

  if (gid < numvertices)
  {
    float dispX = 0;
    float dispY = 0;

#ifdef YIFAN_HU
    // disp = step * (f / ||f||)
    float step = gGlobalVars[2];
    float vertexForceX = forceX[gid];
    float vertexForceY = forceY[gid];
    dispX = step * (vertexForceX / sqrt(vertexForceX * vertexForceX +
        vertexForceY * vertexForceY));
    dispY = step * (vertexForceY / sqrt(vertexForceX * vertexForceX +
        vertexForceY * vertexForceY));
#else
    dispX = forceX[gid] * speed[gid];
    dispY = forceY[gid] * speed[gid];
#endif

    // Update vertex locations based on speed.
    fa2UpdateLocation(gid, numvertices, vxLocs, vyLocs, dispX, dispY);

    // Set current forces as old forces in vertex data.
    fa2SaveOldForces(gid, numvertices, forceX[gid], forceY[gid], oldForceX,
        oldForceY);
  }
}

/*!
 * Allocated general memory on the device that will be used when running
 * force atlas 2.
 *
 * \param[in,out] data A valid struct where the pointers need to be saved.
 * \param[in] numvertices the number of vertices in the graph.
 */
void fa2PrepareGeneralMemory(ForceAtlas2Data* data, unsigned int numvertices)
{
  // Allocate data for vertices, edges, and fa2 data.
  cudaMalloc(&data->tra, numvertices * sizeof(float));
  cudaMalloc(&data->swg, numvertices * sizeof(float));
  cudaMalloc(&data->speed, numvertices * sizeof(float));
  cudaMalloc(&data->forceX, numvertices * sizeof(float));
  cudaMalloc(&data->forceY, numvertices * sizeof(float));
  cudaMalloc(&data->oldForceX, numvertices * sizeof(float));
  cudaMalloc(&data->oldForceY, numvertices * sizeof(float));
  cudaMalloc(&data->graphSpeed, sizeof(float));
}

/*!
 * Prepares all memory to run force atlas 2 on the device.
 *
 * \param[in,out] data A valid data object where all pointers should be saved.
 * \param[in] numvertices The total number of vertices.
 */
void fa2PrepareMemory(ForceAtlas2Data* data,
    unsigned int numvertices)
{
  fa2PrepareGeneralMemory(data, numvertices);
}

/*!
 * Cleans the general memory that is required for force atlas 2 on the device.
 * This includes vertex data. This excludes edge data.
 *
 * \param[in] data The struct that holds the data pointers.
 */
void fa2CleanGeneralMemory(ForceAtlas2Data* data)
{
  cudaFree(data->tra);
  cudaFree(data->swg);
  cudaFree(data->speed);
  cudaFree(data->forceX);
  cudaFree(data->forceY);
  cudaFree(data->oldForceX);
  cudaFree(data->oldForceY);
  cudaFree(data->graphSpeed);
}

/*!
 * Cleans the memory on the device that is required for running force atlas 2.
 *
 * \param[in] data The data that needs to be cleaned.
 * \param[in] numvertices The total number of vertices.
 */
void fa2CleanMemory(ForceAtlas2Data* data, unsigned int numvertices)
{
  fa2CleanGeneralMemory(data);
}

#ifdef YIFAN_HU
void yifanhuUpdateStepLength(float hGlobalVars[])
{
  if (hGlobalVars[0] < hGlobalVars[1])
  {
    hGlobalVars[3]++;
    if (hGlobalVars[3] >= 5)
    {
      hGlobalVars[3] = 0;
      hGlobalVars[2] /= YIFAN_HU_T;
    }
  } else {
    hGlobalVars[3] = 0;
    hGlobalVars[2] *= YIFAN_HU_T;
  }
}
#endif

void checkErrors(unsigned int num)
{
  cudaDeviceSynchronize();
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess)
  {
    printf("Error in kernel %u.\n%s\n", num, cudaGetErrorString(code));
    exit(EXIT_FAILURE);
  }
}

void fa2RunOnGraph(Graph* g, unsigned int iterations)
{
  Timer* timerIteration = timerNew();
  Timer* timer = timerNew();

  // Allocate data for fa2 data.
  ForceAtlas2Data data;
  fa2PrepareMemory(&data, g->vertices->numvertices);

  // Global variables on the host.
  float hGlobalVars[GLOBAL_VARS];
  // Global vars and arrays are used to compute global variables based on
  // reductions on vertex information.
#ifdef YIFAN_HU
  // Energy, Energy^0 (previous iteration)
  unsigned int numGlobalArrays = 2;
#else
  // Graph Swing
  // Graph Traction
  // Node speed
  unsigned int numGlobalArrays = 3;
#endif
  float* gGlobalVars;
  cudaMalloc(&gGlobalVars, GLOBAL_VARS * sizeof(float));
  float** gGlobalArrays;
  cudaMalloc(&gGlobalArrays, numGlobalArrays * sizeof(float*));

  float* vxLocs = g->vertices->vertexXLocs;
  float* vyLocs = g->vertices->vertexYLocs;
  unsigned int* numEdges = g->edges->numedges;
  unsigned int* edgeTargets = g->edges->edgeTargets;

  // Initialize the global variables. step, progress
#ifdef YIFAN_HU
  utilVectorSetByScalar(gGlobalVars, 0, GLOBAL_VARS);
  utilVectorSetByScalar(gGlobalVars, FLT_MAX, 1);
#else
  // Graph swing, traction. Graph speed, speed
  utilVectorSetByScalar(gGlobalVars, 0, GLOBAL_VARS);
#endif

  // Set pointers to global arrays.
#ifdef YIFAN_HU
  cudaMemcpy(&gGlobalArrays[0], &data.forceX, sizeof(float*),
      cudaMemcpyHostToDevice);
  cudaMemcpy(&gGlobalArrays[1], &data.forceY, sizeof(float*),
      cudaMemcpyHostToDevice);
#else // Force Atlas 2
  cudaMemcpy(&gGlobalArrays[0], &data.swg, sizeof(float*),
      cudaMemcpyHostToDevice);
  cudaMemcpy(&gGlobalArrays[1], &data.tra, sizeof(float*),
      cudaMemcpyHostToDevice);
  cudaMemcpy(&gGlobalArrays[2], &data.speed, sizeof(float*),
      cudaMemcpyHostToDevice);
#endif

  unsigned int numblocks = ceil(g->vertices->numvertices / (float) BLOCK_SIZE);
  unsigned int numblocks_reduction = ceil(numblocks / 2.0);

#ifdef DEBUG
  checkErrors(0);
#endif

  for (size_t i = 0; i < iterations; i++)
  {
#ifdef YIFAN_HU
    cudaMemcpy(&gGlobalVars[1], &gGlobalVars[0], sizeof(float),
        cudaMemcpyDeviceToDevice);
    float zero = 0;
    cudaMemcpy(&gGlobalVars[0], &zero, sizeof(float), cudaMemcpyHostToDevice);
#else
#endif

    // Run fa2 spring embedding kernel.
    startTimer(timerIteration);

    // Compute graph speed, vertex forces, speed and displacement.
    startTimer(timer);
    resetForcesKernel<<<numblocks, BLOCK_SIZE>>>(g->vertices->numvertices,
    data.forceX, data.forceY);
    stopTimer(timer);
    printTimer(timer, "time: force: reset.");
    resetTimer(timer);

    startTimer(timer);
    fa2Gravity<<<numblocks, BLOCK_SIZE>>>(g->vertices->numvertices, vxLocs,
    vyLocs, data.forceX, data.forceY, numEdges);
    stopTimer(timer);
    printTimer(timer, "time: force: gravity.");
    resetTimer(timer);

    startTimer(timer);
    fa2Repulsion<<<numblocks, BLOCK_SIZE>>>(g->vertices->numvertices, vxLocs,
    vyLocs, data.forceX, data.forceY, numEdges);
    stopTimer(timer);
    printTimer(timer, "time: force: repulsion.");
    resetTimer(timer);

    startTimer(timer);
    fa2Attraction<<<numblocks, BLOCK_SIZE>>>(g->vertices->numvertices, vxLocs,
    vyLocs, numEdges, edgeTargets, g->edges->maxedges, data.forceX, data.forceY);
    stopTimer(timer);
    printTimer(timer, "time: force: attraction.");
    resetTimer(timer);

#ifdef DEBUG
    checkErrors(1);
#endif

#ifdef YIFAN_HU
#else
    startTimer(timer);
    fa2UpdateSwing<<<numblocks, BLOCK_SIZE>>>(g->vertices->numvertices,
    data.forceX, data.forceY, data.oldForceX, data.oldForceY, data.swg, data.tra);
    stopTimer(timer);
    printTimer(timer, "time: globals: traction and swing.");
    resetTimer(timer);

    // Run reductions on vertex swing and traction.
    // Or in the case of yifan hu: energy
    startTimer(timer);
    fa2GraphSwingTract<<<numblocks_reduction, BLOCK_SIZE>>>(
        g->vertices->numvertices, gGlobalArrays, numEdges,
        gGlobalVars);

    cudaMemcpy(hGlobalVars, gGlobalVars, GLOBAL_VARS * sizeof(float),
      cudaMemcpyDeviceToHost);
    // Update speed of Graph.
    fa2UpdateSpeedGraph(hGlobalVars[0], hGlobalVars[1], &hGlobalVars[2]);
    cudaMemcpy(gGlobalVars, hGlobalVars, GLOBAL_VARS * sizeof(float),
      cudaMemcpyHostToDevice);
    // Update speed of vertices.
    fa2UpdateSpeed<<<numblocks, BLOCK_SIZE>>>(g->vertices->numvertices, gGlobalArrays[2], data.swg,
    data.forceX, data.forceY,
        gGlobalVars);

    stopTimer(timer);
    printTimer(timer, "time: graph swing and traction.");
    resetTimer(timer);
#endif

#ifdef DEBUG
    checkErrors(2);
#endif

    fa2MoveVertices<<<numblocks, BLOCK_SIZE>>>(vxLocs, vyLocs,
        g->vertices->numvertices, gGlobalVars, data.tra, data.swg,
        data.forceX, data.forceY,
        data.oldForceX, data.oldForceY, data.speed);

#ifdef DEBUG
    checkErrors(3);
#endif

#ifdef YIFAN_HU
    startTimer(timer);
    cudaMemcpy(hGlobalVars, gGlobalVars, GLOBAL_VARS * sizeof(float),
      cudaMemcpyDeviceToHost);
    yifanhuUpdateStepLength(hGlobalVars);
    cudaMemcpy(gGlobalVars, hGlobalVars, GLOBAL_VARS * sizeof(float),
      cudaMemcpyHostToDevice);
    stopTimer(timer);
    printTimer(timer, "time: globals: step length.");
    resetTimer(timer);
#else
#endif

    stopTimer(timerIteration);
    printTimer(timerIteration, "time: embedding iteration.");
    resetTimer(timerIteration);
  }

  fa2CleanMemory(&data, g->vertices->numvertices);
}

