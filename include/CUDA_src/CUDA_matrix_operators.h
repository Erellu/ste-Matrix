/**

                     ste::Matrix class

    @brief This CUDA header file contains functions applied on ste::Matrix using a GPU.

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


