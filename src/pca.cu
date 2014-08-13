/*!
 * \file
 */

#include "pca.h"
#include "cublas_v2.h"
#include "cula.h"

void pca(float* d_inMatrix, unsigned int inCols, unsigned int inRows,
    float* d_outMatrix, unsigned int outCols, unsigned int outRows)
{
  char jobu = 'A';
  char jobvt = 'N';

  const float alpha = 1.0f;
  const float beta = 0.0f;

  int min = (inCols < inRows) ? inCols : inRows;
  float* S;
  cudaMalloc(&S, min * sizeof(float));
  float* d_U;
  cudaMalloc(&d_U, inRows * inRows * sizeof(float));
  float* d_VT;
  cudaMalloc(&d_VT, inCols * inCols * sizeof(float));

  culaInitialize();
  culaDeviceSgesvd(jobu, jobvt, inRows, inCols, d_inMatrix, inRows, S, d_U,
      inRows, d_VT, inCols);
  culaShutdown();

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, inRows, inCols, inRows, &alpha,
      d_U, outRows, d_inMatrix, inRows, &beta, d_outMatrix, inRows);
  cublasDestroy(handle);
}
