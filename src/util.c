
void utilVectorSetByScalar(float* dst, float scalar, unsigned int num)
{

}

void utilVectorAdd(float* dst, float* src, unsigned int num)
{

}

void utilVectorAddScalar(float* dst, float scalar, unsigned int num)
{

}

void utilVectorAddInStream(float* dst, float* src, unsigned int num,
    void* stream_ptr)
{

}

void utilVectorMultiply(float* dst, float* src, unsigned int num)
{

}

void utilVectorMultiplyByScalar(float* dst, float scalar, unsigned int num)
{

}

void utilVectorDevide(float* dst, float* src, unsigned int num)
{

}

void utilVectorDevideByScalar(float* dst, float scalar, unsigned int num)
{

}

void utilTreeReduction(float* d_M, unsigned int numelems, float* d_outVal)
{

}

void utilPrintDeviceArray(float* array, unsigned int numelems)
{

}

void* utilDataTransferHostToDevice(void* src, unsigned int numbytes,
    unsigned int freeHostMem)
{

}

void* utilDataTransferDeviceToHost(void* src, unsigned int numbytes,
    unsigned int freeDeviceMem)
{

}

void* utilAllocateData(unsigned int numbytes)
{

}

void utilFreeDeviceData(float* dptr)
{

}

void utilCudaCheckError(void* cudaError_t_ptr, char* msg)
{

}

