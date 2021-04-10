#include "CUDA_matrix_operators.h"

namespace ste {

std::vector<float> CUDA_mult_MAT(const std::vector<float> &data_1 , const size_t &data_1_rows, const size_t &data_1_columns,
                                 const std::vector<float> &data_2 , const size_t &data_2_rows, const size_t &data_2_columns){

    (void)data_2_rows; //This is passed in argument only for clarity.

    cublasHandle_t handle;

    ste_cublas_error_check(cublasCreate(&handle));

    std::vector<float> result(data_1_rows * data_2_columns);

    /*----------------------------------------------------------------------------------------------*/

    float* GPU_data_1 = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_data_1) , data_1.size()*sizeof(float)));
    ste_gpu_error_check(cudaMemcpy(GPU_data_1, data_1.data(), data_1.size()*sizeof(float), cudaMemcpyHostToDevice));

    float* GPU_data_2 = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_data_2),data_2.size()*sizeof(float)));
    ste_gpu_error_check(cudaMemcpy(GPU_data_2, data_2.data(), data_2.size()*sizeof(float), cudaMemcpyHostToDevice));

    float* GPU_result = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_result) , result.size()*sizeof(float)));

    /*----------------------------------------------------------------------------------------------*/

    //cublasSgemm(handle , operation , operation , m , n , k , alpha , A , lda , B , ldb , beta , C , ldc

    //(m X n) * (n X k) -> (m X k)

    //C = (alpha*A) * B + (beta*C)

    constexpr float alpha = 1.f; //Needs to be defined as a variable as it can be either a host or a device pointer (type float* in argument)
    constexpr float beta = 0.f;

    ste_cublas_error_check(
                cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                           data_2_columns , data_1_rows ,data_1_columns,
                           &alpha , GPU_data_2 , data_2_columns,
                           GPU_data_1 , data_1_columns,
                           &beta , GPU_result , data_2_columns)
                );


    ste_gpu_error_check(cudaMemcpy(result.data() , GPU_result , result.size() * sizeof(float) , cudaMemcpyDeviceToHost));

    ste_gpu_error_check(cudaFree(GPU_data_1));

    ste_gpu_error_check(cudaFree(GPU_data_2));

    ste_gpu_error_check(cudaFree(GPU_result));

    ste_cublas_error_check(cublasDestroy_v2(handle));

    return result;

}

std::vector<double> CUDA_mult_MAT(const std::vector<double> &data_1 , const size_t &data_1_rows, const size_t &data_1_columns,
                                  const std::vector<double> &data_2 , const size_t &data_2_rows, const size_t &data_2_columns){

    (void)data_2_rows; //This is passed in argument only for clarity.

    cublasHandle_t handle;

    ste_cublas_error_check(cublasCreate(&handle));

    std::vector<double> result(data_1_rows * data_2_columns);

    /*----------------------------------------------------------------------------------------------*/

    double* GPU_data_1 = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_data_1) , data_1.size()*sizeof(double)));
    ste_gpu_error_check(cudaMemcpy(GPU_data_1, data_1.data(), data_1.size()*sizeof(float), cudaMemcpyHostToDevice));

    double* GPU_data_2 = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_data_2),data_2.size()*sizeof(double)));
    ste_gpu_error_check(cudaMemcpy(GPU_data_2, data_2.data(), data_2.size()*sizeof(float), cudaMemcpyHostToDevice));

    double* GPU_result = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_result) , result.size()*sizeof(double)));

    /*----------------------------------------------------------------------------------------------*/

    //cublasSgemm(handle , operation , operation , m , n , k , alpha , A , lda , B , ldb , beta , C , ldc

    //(m X n) * (n X k) -> (m X k)

    //C = (alpha*A) * B + (beta*C)

    constexpr double alpha = 1.f; //Needs to be defined as a variable as it can be either a host or a device pointer (type float* in argument)
    constexpr double beta = 0.f;

    ste_cublas_error_check(
                cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                           data_2_columns , data_1_rows ,data_1_columns,
                           &alpha , GPU_data_2 , data_2_columns,
                           GPU_data_1 , data_1_columns,
                           &beta , GPU_result , data_2_columns)
                );


    ste_gpu_error_check(cudaMemcpy(result.data() , GPU_result , result.size() * sizeof(float) , cudaMemcpyDeviceToHost));

    ste_gpu_error_check(cudaFree(GPU_data_1));

    ste_gpu_error_check(cudaFree(GPU_data_2));

    ste_gpu_error_check(cudaFree(GPU_result));

    ste_cublas_error_check(cublasDestroy_v2(handle));

    return result;

}


std::vector<float> CUDA_transpose(const std::vector<float> &data , const size_t &rows , const size_t &columns){

    cublasHandle_t handle;
    ste_cublas_error_check(cublasCreate(&handle));

    std::vector<float> result(data.size());

    float* GPU_data = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_data) , data.size()*sizeof(float)));

    float* GPU_data_clone = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_data_clone) , data.size()*sizeof(float)));
    ste_gpu_error_check(cudaMemcpy(GPU_data_clone, data.data(), data.size()*sizeof(float), cudaMemcpyHostToDevice));

    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;

    cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, columns, &alpha, GPU_data_clone, columns , &beta, GPU_data_clone, rows, GPU_data, rows );


    ste_gpu_error_check(cudaMemcpy(result.data() , GPU_data , result.size() * sizeof(float) , cudaMemcpyDeviceToHost));

    ste_gpu_error_check(cudaFree(GPU_data));
    ste_gpu_error_check(cudaFree(GPU_data_clone));

    ste_cublas_error_check(cublasDestroy_v2(handle));


    return result;
}

std::vector<double> CUDA_transpose(const std::vector<double> &data , const size_t &rows , const size_t &columns){

    cublasHandle_t handle;
    ste_cublas_error_check(cublasCreate(&handle));

    std::vector<double> result(data.size());

    double* GPU_data = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_data) , data.size()*sizeof(double)));

    double* GPU_data_clone = NULL;
    ste_gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&GPU_data_clone) , data.size()*sizeof(double)));
    ste_gpu_error_check(cudaMemcpy(GPU_data_clone, data.data(), data.size()*sizeof(double), cudaMemcpyHostToDevice));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    cublasDgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, columns, &alpha, GPU_data_clone, columns , &beta, GPU_data_clone, rows, GPU_data, rows );

    ste_gpu_error_check(cudaMemcpy(result.data() , GPU_data , result.size() * sizeof(double) , cudaMemcpyDeviceToHost));

    ste_gpu_error_check(cudaFree(GPU_data));
    ste_gpu_error_check(cudaFree(GPU_data_clone));

    ste_cublas_error_check(cublasDestroy_v2(handle));

    return result;

}


}//namespace ste




