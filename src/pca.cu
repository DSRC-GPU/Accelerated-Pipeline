/*!
 * \file
 */

#include "pca.h"
#include "cublas_v2.h"
#include "cula.h"
#include <stdio.h>
#include "util.h"

void pca(float* d_inMatrix, unsigned int inRows, unsigned int inCols,
    float* d_outMatrix)
{
  float* d_Y = NULL;
  float* d_PC = NULL;

  cudaMalloc(&d_PC, inCols * inCols * sizeof(float));
  utilVectorSetByScalar(d_PC, 0, inCols * inCols);

  cudaMalloc(&d_Y, inRows * inCols * sizeof(float));
  utilVectorSetByScalar(d_Y, 0, inRows * inCols);

  // Subtract mean for each dimension.
  pcaUpdateMean(d_inMatrix, inRows, inCols);

  // Calculate matrix Y.
  pcaCalculateYMatrix(d_inMatrix, inRows, inCols, d_Y);

  // Perform SVD on Y.
  pcaSVD(d_Y, inCols, inRows, d_PC);

  DEBUG_PRINT_DEVICE(d_PC, inRows * inRows);
  DEBUG_PRINT_DEVICE(d_inMatrix, inRows * inCols);

  // Calculate signals.
  pcaCalculateSignals(d_PC, d_inMatrix, inRows, inCols, d_outMatrix);
}

void pcaUpdateMean(float* d_inMatrix, unsigned int inRows, unsigned int inCols)
{
  float* d_averageX = NULL;
  float* d_averageY = NULL;
  cudaMalloc(&d_averageX, sizeof(float));
  cudaMalloc(&d_averageY, sizeof(float));

  // Compute the average X and Y values.
  utilParallelSum(&d_inMatrix[0], inCols, d_averageX);
  utilParallelSum(&d_inMatrix[inCols], inCols, d_averageY);

  float h_averageX, h_averageY;
  cudaMemcpy(&h_averageX, d_averageX, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_averageY, d_averageY, sizeof(float), cudaMemcpyDeviceToHost);
  h_averageX /= inCols;
  h_averageY /= inCols;

  utilVectorAddScalar(&d_inMatrix[0], -1 * h_averageX, inCols);
  utilVectorAddScalar(&d_inMatrix[inCols], -1 * h_averageY, inCols);
}

void pcaCalculateYMatrix(float* d_inMatrix, unsigned int inRows, unsigned int
    inCols, float* d_Y)
{
  // Transpose inMatrix
  // Do not transpose because CULA expects column major ordering.
  //
  // const float alpha = 1;
  // const float beta = 0;
  // cublasHandle_t handle;
  // cublasCreate(&handle);
  // cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, inCols, inRows, &alpha,
  //     d_inMatrix, inRows, &beta, NULL, inRows, d_Y, inCols);
  // cublasDestroy(handle);
  
  // Devide all values by sqrt(N-1)
  float sqrtN = sqrt(inCols - 1);
  cudaMemcpy(d_Y, d_inMatrix, inRows * inCols * sizeof(float),
      cudaMemcpyDeviceToDevice);
  utilVectorDevideByScalar(d_Y, sqrtN, inRows * inCols); 
}

void pcaSVD(float* d_Y, unsigned int inRows, unsigned int inCols, float* d_PC)
{
  char jobu = 'N';
  char jobvt = 'A';

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

  cudaFree(S);
  cudaFree(d_U);
}

void pcaCalculateSignals(float* d_PC, float* d_inMatrix, unsigned int inRows,
    unsigned int inCols, float* d_Signals)
{
  const float alpha = 1;
  const float beta = 0;
  cublasHandle_t handle;
  cublasCreate(&handle);

  float m = inRows;
  float n = inCols;
  float k = inRows;
  float lda = inRows;
  float ldb = inCols;
  float ldc = inRows;

  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha,
      d_PC, lda, d_inMatrix, ldb, &beta, d_Signals, ldc);

  cublasDestroy(handle);
}

