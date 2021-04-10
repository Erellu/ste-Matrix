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


