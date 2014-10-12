
#include "pca.h"
#include <stdio.h>
#include <dataanalysis.h>

using namespace alglib;

void projectData(real_2d_array inMatrix, real_2d_array v, unsigned int rows,
    unsigned int cols, real_2d_array& outMatrix)
{
  ae_int_t m = rows;
  ae_int_t n = cols;
  ae_int_t k = cols;
  double alpha = 1.0;
  ae_int_t ia = 0;
  ae_int_t ja = 0;
  ae_int_t optypea = 0;
  ae_int_t ib = 0;
  ae_int_t jb = 0;
  ae_int_t optypeb = 0;
  double beta = 0.0;
  ae_int_t ic = 0;
  ae_int_t jc = 0;

  rmatrixgemm(m, n, k, alpha, inMatrix, ia, ja, optypea, v, ib, jb, optypeb, beta,
      outMatrix, ic, jc);
}

void pcaUpdateMean(float* d_inMatrix, unsigned int inRows, unsigned int inCols)
{
  float* averages = (float*) calloc(inRows, sizeof(float));
  for (size_t i = 0; i < inRows; i++)
  {
    for (size_t j = 0; j < inCols; j++)
    {
      averages[i] += d_inMatrix[j + i * inCols];
    }
    averages[i] /= inCols;
  }

  for (size_t i = 0; i < inRows; i++)
  {
    for (size_t j = 0; j < inCols; j++)
    {
      d_inMatrix[j + i * inCols] -= averages[i];
    }
  }
  free(averages);
}

void pca(float* d_inMatrix, unsigned int inRows, unsigned int inCols,
    float* d_outMatrix)
{
  pcaUpdateMean(d_inMatrix, inRows, inCols);

  real_2d_array inMatrix;
  double* inMatrixDouble = (double*) calloc(inRows * inCols, sizeof(double)); 

  for (size_t i = 0; i < inRows * inCols; i++)
  {
    unsigned int row_i = floor(i / inCols);
    unsigned int col_i = i - (row_i * inCols);

    // Begin transpose
    unsigned int tmp = row_i;
    row_i = col_i;
    col_i = tmp;
    // End transpose

    unsigned int index = row_i * inRows + col_i;

    inMatrixDouble[index] = d_inMatrix[i];
  }
  inMatrix.setcontent(inCols, inRows, inMatrixDouble);  

  ae_int_t info;
  real_1d_array s2;
  real_2d_array v;
  pcabuildbasis(inMatrix, inCols, inRows, info, s2, v);

  printf("V\n");
  for (size_t i = 0; i < inRows; i++)
  {
    for (size_t j = 0; j < inRows; j++)
    {
      printf("%lu,%lu: %f\n", i, j, v[i][j]);
    }
  }

  real_2d_array outMatrix;
  outMatrix.setlength(inCols, inRows);
  projectData(inMatrix, v, inCols, inRows, outMatrix);

  for (size_t i = 0; i < inRows; i++)
  {
    for (size_t j = 0; j < inCols; j++)
    {
      d_outMatrix[j + i * inCols] = outMatrix[j][i];
    }
  }

  free(inMatrixDouble);
}

