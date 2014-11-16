
#include "test-pca.h"
#include "pca.h"
#include "stdio.h"

void testPca()
{
  unsigned int m = 2;
  unsigned int n = 10;

  float* h_In;
  float* h_Out;
  h_In = (float*) calloc(m * n, sizeof(float));
  h_Out = (float*) calloc(m * n, sizeof(float));
  for (size_t i = 0; i < m * n; i++)
  {
    h_In[i] = (float) i;
  }
  
  float* d_In;
  float* d_Out;
  cudaMalloc(&d_In, m * n * sizeof(float));
  cudaMalloc(&d_Out, m * n * sizeof(float));
  cudaMemcpy(d_In, h_In, m * n * sizeof(float), cudaMemcpyHostToDevice);

  pca(d_In, m, n, d_Out);

  cudaMemcpy(h_In, d_In, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Out, d_Out, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  printf("&&&\n");
  for (size_t i = 0; i < m * n; i++)
  {
    printf("%f\n", h_In[i]);
  }
  printf("&&&&\n");
  for (size_t i = 0; i < m * n; i++)
  {
    printf("%f\n", h_Out[i]);
  }
}

//void testCalcYMatrix()
//{
//  unsigned int m = 5;
//  unsigned int n = 2;
//
//  float* h_In;
//  float* h_Out;
//  h_In = (float*) calloc(m * n, sizeof(float));
//  h_Out = (float*) calloc(m * n, sizeof(float));
//  for (size_t i = 0; i < m * n; i++)
//  {
//    h_In[i] = (float) i;
//  }
//  
//  float* d_In;
//  float* d_Out;
//  cudaMalloc(&d_In, m * n * sizeof(float));
//  cudaMalloc(&d_Out, m * n * sizeof(float));
//  cudaMemcpy(d_In, h_In, m * n * sizeof(float), cudaMemcpyHostToDevice);
//
//  pcaCalculateYMatrix(d_In, m, n, d_Out);
//
//  cudaMemcpy(h_In, d_In, m * n * sizeof(float), cudaMemcpyDeviceToHost);
//  cudaMemcpy(h_Out, d_Out, m * n * sizeof(float), cudaMemcpyDeviceToHost);
//
//  for (size_t i = 0; i < m * n; i++)
//  {
//    printf("%f\n", h_In[i]);
//  }
//  printf("&&&&\n");
//  for (size_t i = 0; i < m * n; i++)
//  {
//    printf("%f\n", h_Out[i]);
//  }
//}

