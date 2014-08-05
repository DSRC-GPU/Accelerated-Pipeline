/*
 * \file speedvector.h
 */

#ifndef SPEEDVECTOR_H_
#define SPEEDVECTOR_H_

void speedVectorInit(float** averageSpeedX, float** averageSpeedY,
    float* vxLocs, float* vyLocs, unsigned int numvertices);

void speedVectorUpdate(float* vxLocs, float* vyLocs, float* averageSpeedX,
    float* averageSpeedY, unsigned int numvertices, cudaStream_t* stream);

void speedVectorFinish(float* averageSpeedX, float* averageSpeedY,
    unsigned int numiterations, unsigned int numvertices);

void speedVectorClean(float* averageSpeedX, float* averageSpeedY);

#endif /* SPEEDVECTOR_H_ */
