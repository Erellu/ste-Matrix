#include "CUDA_setup.h"

#ifdef __cplusplus
extern "C"{
#endif

bool CUDA_setup(){

    int devCount;
    cudaGetDeviceCount(&devCount);

    return (devCount > 0);
}

#ifdef __cplusplus
}
#endif
