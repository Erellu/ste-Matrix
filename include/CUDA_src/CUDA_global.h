#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>

#include <iostream>

namespace ste {


    /**
        @brief Convenience function to display a 'cublasStatus_t' error code in case of failure. Should be called through the macro ste_cublas_error_check.

        @arg code: cublasStatus_t error code
        @arg file: file in which the function is called. Automatically determined by the macro ste_cublas_error_check.
        @arg line: line at which the function is called. Automatically determined by the macro ste_cublas_error_check.
    */
    extern void CUBLAs_assert(const cublasStatus_t &code, const char *file, const size_t &line);

    /**
        @brief Convenience function to display a 'cudaError_t' error code in case of failure. Should be called through the macro ste_gpu_error_check.

        @arg code: cublasStatus_t error code
        @arg file: file in which the function is called. Automatically determined by the macro ste_gpu_error_check.
        @arg line: line at which the function is called. Automatically determined by the macro ste_gpu_error_check.
    */
    extern void gpu_assert(const cudaError_t &code, const char *file,  const size_t &line);

} //namespace ste

#define ste_cublas_error_check(ans) { ste::CUBLAs_assert((ans), __FILE__, __LINE__); }
#define ste_gpu_error_check(ans) { ste::gpu_assert((ans), __FILE__, __LINE__); }

