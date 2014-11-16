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

#endif /* PCA_H_ */
