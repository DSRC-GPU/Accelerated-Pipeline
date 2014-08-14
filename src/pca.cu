/*!
 * \file
 */

#include "pca.h"
#include "cublas_v2.h"
#include "cula.h"
#include "stdio.h"
#include "util.h"

float* pca(float* d_inMatrix, unsigned int inCols, unsigned int inRows,
    float* d_outMatrix)
{
  float* d_Y;
  float* d_PC;
  float* d_Signals;

  cudaMalloc(&d_PC, inCols * inCols * sizeof(float));
  utilVectorSetByScalar(d_PC, 0, inCols * inCols);

  // Subtract mean for each dimension.
  pcaUpdateMean(d_inMatrix, inRows, inCols);

  // Calculate matrix Y.
  pcaCalculateYMatrix(d_inMatrix, inRows, inCols, d_Y);

  // Perform SVD on Y.
  pcaSVD(d_Y, inRows, inCols, d_PC);
  // TODO Wrap code below in a function call.

  float* h_inMatrix = (float*) calloc(inCols * inRows, sizeof(float));
  h_inMatrix[0] = 1.0;
  h_inMatrix[7] = 4.0;
  h_inMatrix[9] = 3.0;
  h_inMatrix[16] = 2.0;
  cudaMalloc(&d_inMatrix, inCols * inRows * sizeof(float));
  cudaMemcpy(d_inMatrix, h_inMatrix, inCols * inRows * sizeof(float),
      cudaMemcpyHostToDevice);
  //utilVectorSetByScalar(d_inMatrix, 1, inCols * inRows);

  for (size_t i = 0; i < inRows * inCols; i++)
  {
    printf("%f\n", h_inMatrix[i]);
  }
  printf("&&&\n");

  // Calculate signals.
  pcaCalculateSignals(d_PC, d_inMatrix, inRows, inCols, d_Signals);

  // Return signals
  return d_Signals;
}

void pcaUpdateMean(float* d_inMatrix, unsigned int inRows, unsigned int inCols)
{

}

void pcaCalculateYMatrix(float* d_inMatrix, unsigned int inRows, unsigned int
    inCols, float* d_Y)
{

}

void pcaSVD(float* d_Y, unsigned int inRows, unsigned int inCols, float* d_PC)
{
  char jobu = 'N';
  char jobvt = 'A';

  const float alpha = 1.0f;
  const float beta = 0.0f;

  int min = (inCols < inRows) ? inCols : inRows;
  float* S;
  cudaMalloc(&S, min * sizeof(float));
  utilVectorSetByScalar(S, 0, min);
  float* d_U;
  cudaMalloc(&d_U, inRows * inRows * sizeof(float));
  utilVectorSetByScalar(d_U, 0, inRows * inRows);

  culaInitialize();
  culaDeviceSgesvd(jobu, jobvt, inRows, inCols, d_Y, inRows, S, d_U,
      inRows, d_PC, inCols);
  culaShutdown();

  // TODO Free some memory.
  cudaFree(S);
  cudaFree(d_U);
}

void pcaCalculateSignals(float* d_PC, float* d_inMatrix, unsigned int inRows,
    unsigned int inCols, float* d_Signals)
{

}

