/**

                     ste::Matrix class

    @brief This CUDA header file contains convenience functions for assertions in CUDA.

    @copyright     Copyright (C) <2020-2021>  DUHAMEL Erwan

                        BSD-2 License

    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice,
          this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


    @authors DUHAMEL Erwan (erwanduhamel@outlook.com)            -- Developper / Tester
             SOUDIER Jean  (jean.soudier@insa-strasbourg.fr)     -- Tester
*/



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

