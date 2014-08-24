/*!
 * \file
 */

#ifndef PCA_H_
#define PCA_H_

/*!
 * Computes the pca of an input values. Matrix should be row-major.
 *
 * \param[in] d_inMatrix The matrix to use as pca input.
 * \param[in] inRows The number of rows in the matrix.
 * \param[in] inCols The number of columns in the matrix.
 * \param[out] d_outMatrix The projected data.
 */
void pca(float* d_inMatrix, unsigned int inRows, unsigned int inCols,
    float* d_outMatrix);

/*!
 * Normalizes the input matrix. Used internally by the pca function.
 *
 * \param[in,out] d_inMatrix The matrix whose values to normalize.
 * \param[in] inRows The number of rows of the matrix.
 * \param[in] inCols The number of columns of the matrix.
 */
void pcaUpdateMean(float* d_inMatrix, unsigned int inRows, unsigned int inCols);

/*!
 * Calculates the Y matrix. Used iternally by the pca function.
 *
 * \param[in] d_inMatrix The input matrix.
 * \param[in] inRows The number of rows in the input matrix.
 * \param[in] inCols The number of columns in the input matrix.
 * \param[out] d_Y The output Y matrix.
 */
void pcaCalculateYMatrix(float* d_inMatrix, unsigned int inRows, unsigned int
    inCols, float* d_Y);

/*!
 * Single Value Decomposition used by the pca function.
 *
 * \param[in] d_Y The Y matrix.
 * \param[in] inRows The number of rows in the Y matrix.
 * \param[in] inCols The number of columns in the Y matrix.
 * \param[out] d_PC The principal component matrix.
 */
void pcaSVD(float* d_Y, unsigned int inRows, unsigned int inCols, float* d_PC);

/*!
 * Calculates the signals, or 'projected data'.
 *
 * \param[in] d_PC The principal component matrix.
 * \param[in] d_inMatrix The input matrix.
 * \param[in] inRows The number of rows in the PC and input matrix.
 * \param[in] inCols The number of columns in the PC and input matrix.
 * \param[out] d_Signals The projected data.
 */
void pcaCalculateSignals(float* d_PC, float* d_inMatrix, unsigned int inRows,
    unsigned int inCols, float* d_Signals);

#endif /* PCA_H_ */
