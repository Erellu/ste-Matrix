#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

//#include <cublas.h>

#include "CUDA_global.h"

#include <vector>
#include <iostream>

extern std::vector<float> CUDA_mult_MAT(const std::vector<float> &data_1 , const uint64_t data_1_rows, const uint64_t data_1_columns,
                          const std::vector<float> &data_2 , const uint64_t data_2_rows, const uint64_t data_2_columns);

extern std::vector<double> CUDA_mult_MAT(const std::vector<double> &data_1 , const uint64_t data_1_rows, const uint64_t data_1_columns,
                          const std::vector<double> &data_2 , const uint64_t data_2_rows, const uint64_t data_2_columns);


/*
extern __global__ void CUDA_mult_MAT_KERNEL(float *return_value , const uint64_t return_rows, const uint64_t return_columns,
                                            const float *data_1 ,
                                           const float *data_2 );
*/
/*extern void CUDA_mult_T(const std::vector<float> &data , const uint64_t data_rows, const uint64_t data_columns,
                        const float value , 
                        std::vector<float> &result);*/

/*
extern __global__ void CUDA_mult_T_KERNEL(float* return_value ,
                                          const float *data_1  ,
                                          const float value
                                         );
*/
