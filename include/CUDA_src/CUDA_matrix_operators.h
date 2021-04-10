#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "CUDA_global.h"

#include <vector>
#include <iostream>


namespace ste {

    /**
        @brief Determines the product of two matrices A*B.

        @arg data_1: data of the left matrix.
        @arg data_1_rows : rows of the left matrix.
        @arg data_1_columns : columns of the left matrix.
        @arg data_2: data of the right matrix.
        @arg data_2_rows : rows of the right matrix.
        @arg data_2_columns : columns of the right matrix.

        @return Product of the two matrices stored in a std::vector. Output matrix size is managed by ste::Matrix operators.
    */
    extern std::vector<float> CUDA_mult_MAT(const std::vector<float> &data_1 , const size_t &data_1_rows, const size_t &data_1_columns,
                                            const std::vector<float> &data_2 , const size_t &data_2_rows, const size_t &data_2_columns);

    /**
        @brief Determines the product of two matrices A*B.

        @arg data_1: data of the left matrix.
        @arg data_1_rows : rows of the left matrix.
        @arg data_1_columns : columns of the left matrix.
        @arg data_2: data of the right matrix.
        @arg data_2_rows : rows of the right matrix.
        @arg data_2_columns : columns of the right matrix.

        @return Product of the two matrices stored in a std::vector. Output matrix size is managed by ste::Matrix operators.
    */
    extern std::vector<double> CUDA_mult_MAT(const std::vector<double> &data_1 , const size_t &data_1_rows, const size_t &data_1_columns,
                                             const std::vector<double> &data_2 , const size_t &data_2_rows, const size_t &data_2_columns);


    /**********************************************************************/


    /**
        @brief Computes the transpose of a matrix.

        @arg data: data of the matrix.
        @arg rows : rows of the matrix.
        @arg columns : columns of the matrix.

        @return Transpose of the matrix stored in a std::vector. Output matrix size is managed by ste::Matrix functions.
    */
    extern std::vector<float> CUDA_transpose(const std::vector<float> &data , const size_t &rows , const size_t &columns);

    /**
        @brief Computes the transpose of a matrix.

        @arg data: data of the matrix.
        @arg rows : rows of the matrix.
        @arg columns : columns of the matrix.

        @return Transpose of the matrix stored in a std::vector. Output matrix size is managed by ste::Matrix functions.
    */
    extern std::vector<double> CUDA_transpose(const std::vector<double> &data , const size_t &rows , const size_t &columns);
}


