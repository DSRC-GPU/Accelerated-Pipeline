/*!
 * \file
 */

#include "pca.h"
#include "cublas_v2.h"
#include "cula.h"
#include "stdio.h"
#include "util.h"

void pca(float* d_inMatrix, unsigned int inCols, unsigned int inRows,
    float* d_outMatrix, unsigned int outCols, unsigned int outRows)
{
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

  char jobu = 'A';
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
  float* d_VT;
  cudaMalloc(&d_VT, inCols * inCols * sizeof(float));
  utilVectorSetByScalar(d_VT, 0, inCols * inCols);

  culaInitialize();
  culaDeviceSgesvd(jobu, jobvt, inRows, inCols, d_inMatrix, inRows, S, d_U,
      inRows, d_VT, inCols);
  culaShutdown();

  float* h_U = (float*) calloc(inRows * inRows, sizeof(float));
  cudaMemcpy(h_U, d_U, inRows * inRows * sizeof(float),
      cudaMemcpyDeviceToHost);
  float* h_S = (float*) calloc(min, sizeof(float));
  cudaMemcpy(h_S, S, min * sizeof(float), cudaMemcpyDeviceToHost);
  float* h_VT = (float*) calloc(inCols * inCols, sizeof(float));
  cudaMemcpy(h_VT, d_VT, inCols * inCols * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < inRows * inRows; i++)
  {
    printf("%f\n", h_U[i]);
  }
  printf("&&&\n");
  for (size_t i = 0; i < min; i++)
  {
    printf("%f\n", h_S[i]);
  }
  printf("&&&\n");
  for (size_t i = 0; i < inCols * inCols; i++)
  {
    printf("%f\n", h_VT[i]);
  }

  return;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, inRows, inCols, inRows, &alpha,
      d_U, outRows, d_inMatrix, inRows, &beta, d_outMatrix, inRows);
  cublasDestroy(handle);
}
