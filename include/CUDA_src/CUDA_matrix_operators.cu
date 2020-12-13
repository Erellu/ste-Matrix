#include "CUDA_matrix_operators.h"

std::vector<float> CUDA_mult_MAT(const std::vector<float> &data_1 , const uint64_t data_1_rows, const uint64_t data_1_columns,
                                 const std::vector<float> &data_2 , const uint64_t data_2_rows, const uint64_t data_2_columns){

    cublasHandle_t handle;

    cublasErrchk(cublasCreate(&handle));

//    std::cout << "data_1_rows: " << data_1_rows << " data_1_columns: " << data_1_columns << "\n";
//    std::cout << "data_2_rows: " << data_2_rows << " data_2_columns: " << data_2_columns << "\n";


    std::vector<float> result(data_1_rows * data_2_columns);

    /*----------------------------------------------------------------------------------------------*/

    float* GPU_data_1 = NULL;
    gpuErrchk(cudaMalloc((void**)&GPU_data_1 , data_1.size()*sizeof(float)));
    gpuErrchk(cudaMemcpy(GPU_data_1, data_1.data(), data_1.size()*sizeof(float), cudaMemcpyHostToDevice));

    float* GPU_data_2 = NULL;
    gpuErrchk(cudaMalloc((void**)&GPU_data_2 ,data_2.size()*sizeof(float)));
    gpuErrchk(cudaMemcpy(GPU_data_2, data_2.data(), data_2.size()*sizeof(float), cudaMemcpyHostToDevice));

    float* GPU_result = NULL;
    gpuErrchk(cudaMalloc((void**)&GPU_result , result.size()*sizeof(float)));

    /*----------------------------------------------------------------------------------------------*/

    //cublasSgemm(handle , operation , operation , m , n , k , alpha , A , lda , B , ldb , beta , C , ldc

    //(m X n) * (n X k) -> (m X k)

    //C = (alpha*A) * B + (beta*C)

    const float alpha = 1.f; //Needs to be defined as a variable as it can be either a host or a device pointer (type float* in argument)
    const float beta = 0.f;

    cublasErrchk(
                cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                           data_2_columns , data_1_rows ,data_1_columns,
                           &alpha , GPU_data_2 , data_2_columns,
                           GPU_data_1 , data_1_columns,
                           &beta , GPU_result , data_2_columns)
                );


    gpuErrchk(cudaMemcpy(result.data() , GPU_result , result.size() * sizeof(float) , cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(GPU_data_1));

    gpuErrchk(cudaFree(GPU_data_2));

    gpuErrchk(cudaFree(GPU_result));

    cublasErrchk(cublasDestroy_v2(handle));



    return result;


}



