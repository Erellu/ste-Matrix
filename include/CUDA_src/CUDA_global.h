#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>



#include <iostream>

extern void gpuAssert(cudaError_t code, const char *file, int line);

//extern void cublasErrchk(cublasStatus_t code);
extern void cublasAssert(cublasStatus_t code, const char *file, int line);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }

#define ThreadsPerBlock_DIFFERENCE 256
#define ThreadsPerBlock_DOT 512


//extern void gpuErrchk(cudaError_t code);

//extern constexpr int threadsPerBlock = 256;


