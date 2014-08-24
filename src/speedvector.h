/*!
 * \file
 */

#ifndef SPEEDVECTOR_H_
#define SPEEDVECTOR_H_

/*!
 * Initialize speed vector data on the device.
 *
 * \param[out] averageSpeedX Will hold a pointer to the average x speed vector array on the device.
 * \param[out] averageSpeedY Will hold a pointer to the average y speed vector array on the device.
 * \param[in] vxLocs The x location of the vertices on the device.
 * \param[in] vyLocs The y location of the vertices on the device.
 * \param[in] numvertices The total number of vertices.
 */
void speedVectorInit(float** averageSpeedX, float** averageSpeedY,
    float* vxLocs, float* vyLocs, unsigned int numvertices);

/*!
 * Updates the average speed vectors after one run of the spring embedding.
 *
 * \param[in] vxLocs The x positions of the vertices on the device.
 * \param[in] vyLocs The y positions of the vertices on the device.
 * \param[in,out] averageSpeedX The average speeds in the x direction on the device.
 * \param[in,out] averageSpeedY The average speeds in the y direction on the device.
 * \param[in] numvertices The total number of vertices.
 * \param[in] stream The cuda stream to use for these operations.
 */
void speedVectorUpdate(float* vxLocs, float* vyLocs, float* averageSpeedX,
    float* averageSpeedY, unsigned int numvertices, void* stream_ptr);

/*!
 * Completes the average speed vectors.
 *
 * \param[in,out] averageSpeedX The average speeds in the x directions on the device.
 * \param[in,out] averageSpeedY The average speeds in the y directions on the device.
 * \param[in] numiterations The number of runs of the springembedding that were used,
 * should be equal to the number of times 'speedVectorUpdate' has been run.
 * \param[in] numvertices The total number of vertices in the graph.
 */
void speedVectorFinish(float* averageSpeedX, float* averageSpeedY,
    unsigned int numiterations, unsigned int numvertices);

/*!
 * Cleans the memory that was used for these operations.
 *
 * \param[in] averageSpeedX The average speeds in the x directions on the device.
 * \param[in] averageSpeedY The average speeds in the y directions on the device.
 */
void speedVectorClean(float* averageSpeedX, float* averageSpeedY);

#endif /* SPEEDVECTOR_H_ */
