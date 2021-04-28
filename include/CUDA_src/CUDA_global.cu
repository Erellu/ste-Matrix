/**

                     ste::Matrix class

    @brief This CUDA source file contains convenience functions for assertions in CUDA.

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


#include "CUDA_global.h"

namespace ste {

void CUBLAs_assert(const cublasStatus_t &code, const char *file, const size_t &line){

    if(code != CUBLAS_STATUS_SUCCESS){
        std::cerr <<  "CUBLAS error.\nError code: ";

        switch(code){
            case CUBLAS_STATUS_SUCCESS:{std::cerr << "CUBLAS_STATUS_SUCCESS."; break;}

            case CUBLAS_STATUS_NOT_INITIALIZED:{std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED."; break;}

            case CUBLAS_STATUS_ALLOC_FAILED:{std::cerr << "CUBLAS_STATUS_ALLOC_FAILED."; break;}

            case CUBLAS_STATUS_INVALID_VALUE:{std::cerr << "CUBLAS_STATUS_INVALID_VALUE."; break;}

            case CUBLAS_STATUS_ARCH_MISMATCH:{std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH."; break;}

            case CUBLAS_STATUS_MAPPING_ERROR:{std::cerr << "CUBLAS_STATUS_MAPPING_ERROR."; break;}

            case CUBLAS_STATUS_EXECUTION_FAILED:{std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED."; break;}

            case CUBLAS_STATUS_INTERNAL_ERROR:{std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR."; break;}

            case CUBLAS_STATUS_NOT_SUPPORTED:{std::cerr << "CUBLAS_STATUS_NOT_SUPPORTED."; break;}

            case CUBLAS_STATUS_LICENSE_ERROR:{std::cerr << "CUBLAS_STATUS_LICENSE_ERROR."; break;}

            default:{std::cerr << "<unknown>."; break;}

        }

        std::cerr << "\n  File: "<< file << "\n  Line: "<< line <<std::endl;

        exit(EXIT_FAILURE);

    }


}




void gpu_assert(const cudaError_t &code, const char *file, const size_t &line){

   if (code != cudaSuccess){
       std::cerr << "ste::gpu_assert failed.\n  Error: " << cudaGetErrorString(code) << "\n  File: " << file << "\n  Line: " << line << std::endl;
   }

}

} //namespace ste



//void gpuErrchk(cudaError_t code){gpuAssert(code , __FILE__ , __LINE__);}


