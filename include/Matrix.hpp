#ifndef MATRIX_H
#define MATRIX_H

//#define USE_GPU


///TODO / Upcoming :

///     * CUDA_det                      -- Determines the determinant using the GPU
///     * CUDA_cofactormatrix           -- Determines the cofactor matrix using the GPU
///     * CUDA_transpose                -- Determines the transpose matrix using the GPU
///     * CUDA_invert // CUDA_inverse   -- Determines the inverse using the GPU


/**

                     Matrix class

    DUHAMEL Erwan (erwanduhamel@outlook.com)
    SOUDIER Jean  (jean.soudier@insa-strasbourg.fr)


    Provides a template class for matrix-based computing.


    -----------------------------------------------------------------

    Features :

    • Can hold any class.
    • Possibility to use GPU for calculations [WIP]

    • Fast conversion to std::vector<T> to facilitate GPU-acceleration-based algorithms.


    • Operators * and + available as long as the template parameters provides these operators.
    • Operator - avaible for any type that accepts (-1.) as parameter in its constructor.

    • Determinant, inverse, tranpose ,cofactormatrix, trace, mean, average.
    • Classic fill, zeroes, ones, eye, randn and rand.
    • Dynamic resizing (possibility to add, remove and inverst lines and / or columns)
    • Fast reshaping.


    • Possibility to directly print contents and size to stdout.

    • Possibility to override most functions in subclasses to increase performances in specific cases.

    • Convenience types:

          -> Matrix<float>                 <---->    FMatrix
          -> Matrix<double>                <---->    DMatrix
          -> Matrix<long double>           <---->    LDMatrix

          -> Matrix<int>                   <---->    IMatrix
          -> Matrix<long int>              <---->    LIMatrix
          -> Matrix<long long int>         <---->    LLIMatrix

          -> Matrix<unsigned int>          <---->    UIMatrix
          -> Matrix<unsigned long>         <---->    ULMatrix
          -> Matrix<unsigned long long>    <---->    ULLMatrix

          -> Matrix<char>                  <---->    CMatrix
          -> Matrix<unsigned char>         <---->    UCMatrix

    -----------------------------------------------------------------

    Functions :

    Virtual functions are marqued [v].
    Static functions are marqued [S].


    • Matrix            | Constructor. Accepts either a size (x , y or same for both) or a std::vector<std::vector<T>> to construct a Matrix.

    • size              | Returns the size of the matrix, as const std::vector<uint64_t>.
    • columns           | Returns the number of columns of the matrix.
    • rows              | Returns the number of rows of the matrix.
    • lines             | Alias for 'rows'.
    • elements          | Returns the total number of elements in the matrix.
    • clear             | Clears all the element in the matrix, and sets its size to (0 ; 0). WARNING : MEMORY IS NOT FREED.
    • deleteAll         | Calls 'delete' on every element, and sets the matrix size to (0 ; 0). Undefined behavior if T is not a pointer.

    • isRow             | Returns true if the matrix is row, false otherwise.
    • isLine            | Alias for 'isRow'.
    • isColumn          | Returns true if the matrix is a column, false otherwise.
    • isSquare          | Returns true if the matrix is square, false otherwise.
    • isInvertible      | Returns true if the matrix is invertible, false otherwise.
    • empty             | Returns true if the matrix is empty, false otherwise.

    • at                | Returns the element at the index specified in argument. It is passed by reference when the matrix is non-const. Linear indexes can be used.
    • rowAt             | Returns the 'row' at the specified index. It is passed by reference when the matrix is non-const.
    • lineAt            | Alias for 'rowAt'.
    • columnAt          | Returns the column at the specified index. It is always passed by value.

[v] • replace           | Replaces the element specified in argument first (and second if non-linear mode is chosen) by the value in last argument. WARNING ! If T is dynamically allocated, memory IS NOT freed.

[v] • add               | Adds either a line or a column from a vector at the end of the matrix.
    • add_row           | Convenience function to add a row at the end of the matrix.
    • add_line          | Alias for 'add_row'.
    • add_column        | Convenience function to add a column at the end of the matrix.

    • push_back         | Alias for 'add'.
    • push_back_row     | Alias for 'add_row'.
    • push_back_line    | Alias for 'add_line'.
    • push_back_column  | Alias for 'add_column'.

[v] • remove            | Removes either a line or a column at the position specified. WARNING ! If T is dynamically allocated, memory IS NOT freed.
    • remove_row        | Convenience function to remove a row at a specified position. WARNING ! If T is dynamically allocated, memory IS NOT freed.
    • remove_line       | Alias for 'remove_row'. WARNING ! If T is dynamically allocated, memory IS NOT freed.
    • remove_column     | Convenience function to remove a column at a specified position. WARNING ! If T is dynamically allocated, memory IS NOT freed.

[v] • insert            | Inserts either a line or a column at the position specified.
    • insert_row        | Convenience function to insert a line at a specified position.
    • insert_line       | Alias for 'insert_row'.
    • insert column     | Convenience function to insert a column at a specified position.

[v] • swap              | Swaps two lines or two columns at the positions specified.
    • swap_row          | Convenience function to swap two rows at a specified positions.
    • swap_line         | Alias for 'swap_row'.
    • swap_column       | Convenience function to swap two columns at a specified positions.

    • reshape           | Changes the matrix size to the one specified in argument. Throws an exception if the total number of elements in the new size does not match the current one.

[v] • toVector2D        | Converts the matrix to std::vector<std::vector<T>>.
[v] • toVector1D        | Converts the matrix to std::vector<T>

[v] • print             | Prints the contents of the matrix in stdout.
[v] • print_size        | Prints the size of the matrix in stdout.

[v] • begin_row         | Convenience function that returns 0, to provide syntax as close as the one relative to std::algorithm as possible.
    • begin_line        | Alias for 'begin_row'.
[v] • begin_column      | Convenience function that returns 0, to provide syntax as close as the one relative to std::algorithm as possible.
[v] • end_row           | Convenience function that returns the number of lines, to provide syntax as close as the one relative to std::algorithm as possible
    • end_line          | Alias for 'end_row'.
[v] • end_column        | Convenience function that returns the number of columns, to provide syntax as close as the one relative to std::algorithm as possible.

    • for_each          | Analog to std::for_each. Applies the function in argument to every element in the matrix.                                                     ******** WIP : begin_row / end_row / begin_column / end_column
    • transform         | Analog to std::transform. Applies the function in argument to every element in the matrix and modifies them according to its return value.    ******** WIP : begin_row / end_row / begin_column / end_column

[v] • sum               | Returns the sum of all elements of the matrix, as T (meaning that overflow may occur).
[v] • mean              | Returns the mean value of all elements of the matrix, as T (meaning that rounding error and overflow may occur). It is computed as sum()/(rows()*columns().
[v] • average           | Alias for mean().
[v] • max               | Returns the maximum element (according to std::max_element) of the matrix.
[v] • min               | Returns the minimum element (according to std::min_element) of the matrix.

[v] • trace             | Returns the trace of the matrix, computed as T (meaning that rounding error and overflow may occur). Throws an exception (std::invalid_argument) if the matrix is not square.
[v] • det               | Returns the determinant of the matrix. Throws an exception (std::invalid_argument) is the matrix is not square.
[v] • cofactor          | Returns the cofactor of the specified line and column or linear index. Throws an exception if one of them is outside the matrix.
[v] • comatrix          | Returns the cofactor matrix. Convenience function that returns cofactormatrix().
[v] • cofactormatrix    | Returns the cofactor matrix.
[v] • transpose         | Returns the transpose of the matrix.
    • inv               | Returns the inverse of the matrix as computed by operator!.
    • inverse           | Returns the inverse of the matrix as computed by operator!.
[S] • invert            | Returns the inverse of the matrix as computed by operator!.


[v] • hadamard          | Returns the Hadamard product of two matrices. Throws an exception if the sizes do not match.
    • element_wise      | Convenience function that returns the Hadamard product of two matrices.

[v] • fill              | Resizes the matrix as specified in argument and fills it with the value chosen.
[S] • zeroes            | Returns a matrix of the specified dimensions and filled with T(0).
[S] • ones              | Returns a matrix of the specified dimensions and filled with T(1).
[S] • eye               | Returns the identity matrix of the size specified in argument.
[S] • randn             | Creates a matrix of normally distributed numbers.
[S] • uniform           | Creates a matrix of uniformally distributed real numbers.
[S] • uniform_int       | Creates a matrix of uniformally distributed integers.
[S] • rand              | Alias for 'uniform'. Creates a matrix of uniformally distributed numbers.


[v] • operator=         | Assignment operator. Supports assignments from std::vector<std::vector<T>> and from other Matrix.

[v] • operator+         | Computes the addition of a Matrix with another (term by term). Also supports the addition of value in T. In that case, it is the same as adding a Matrix of the same size holding only the same value.
[v] • operator+=        | Adds arg to all the elements of the Matrix, and returns a reference to it.

[v] • operator*         | Computes the usual matrix product of two matrices or multiplies term by term by the T speficied.
[v] • operator*=        | Multiplies the matrix by the argument using the usual matrux product definition (or term by term if the argument is a T), and returns a reference to it.

[v] • operator-         | Computes the substraction of two Matrix or substacts the T argument to all elements.
[v] • operator-=        | Computes the term by term difference of the matrix and the argument, or substracts the T in argument to all elements. Returns a reference to the current item.

[v] • operator!         | Returns the inverse of the matrix.

[v] • operator^         | Returns the matrix two the power specifed after the ^ (ex: a^2 returns a*a);
[v] • operator^=        | Raises the matrix to the specified power, and returns a reference to it.

[v] • operator==        | Equality operator. Returns true only if all elements are identical and at the same position.
[v] • operator!=        | Returns the opposite of the result given by operator==.

*/


#ifdef USE_GPU
#include "CUDA_src/CUDA_setup.h"
#include "CUDA_src/CUDA_matrix_operators.h"
#endif

#include <algorithm>

#include <cmath>

#include <functional>

#include <iostream>
#include <iterator>

#include <random>

#include <stdint.h>
#include <stdexcept>
#include <string>

#include <time.h>

#include <vector>

namespace ste{

template<class T>
class Matrix{

    protected:

       std::vector<T> _data;
       uint64_t _rows;
       uint64_t _columns;

    public:

        /***************************************************/

        enum class Orientation{
            LINE = 0,
            ROW = LINE,
            RW = LINE,
            R = LINE,

            COLUMN = 1,
            COL = COLUMN,
            CL = COLUMN,
            C = COLUMN
        };

        /***************************************************/

        ///Constructor                                                                                | Inits the matrix with empty vectors.
        Matrix(const uint64_t &rows , const uint64_t&columns , const T &value = T(0)){

            #ifdef USE_GPU
            if(!CUDA_setup()){throw std::runtime_error("ste::Matrix::Matrix\nUnable to setup GPU.");}
            #endif

            _rows = rows;
            _columns = columns;
            _data.resize(rows*columns , value);

        }
        ///Constructor                                                                                | Creates an empty square matrix of dimension size.
        Matrix(const uint64_t &size = 0 , const T &value = T(0)) : Matrix(size , size , value){}

        ///Constructor                                                                                | Inits the data of the matrix using a std::vector<std::vector<T>>
        Matrix(const std::vector<std::vector<T>> &data){

            #ifdef USE_GPU
            if(!CUDA_setup()){throw std::runtime_error("ste::Matrix::Matrix\nUnable to setup GPU.");}
            #endif

            const uint64_t column_length = data.at(0).size();

            for(auto &line:data){
                if(line.size() != column_length){throw std::invalid_argument("ste::Matrix::Matrix\nCannot construct a matrix with irregular column size.");}
            }

            _rows = data.size();
            _columns = data.at(0).size();
            _data.reserve(rows() * columns());

            for(auto &line:data){
                _data.insert(_data.end() , line.begin() , line.end());
            }

        }

        Matrix(const std::vector<T> &data , const uint64_t &rows , const uint64_t&columns){

            #ifdef USE_GPU
            if(!CUDA_setup()){throw std::runtime_error("ste::Matrix::Matrix\nUnable to setup GPU.");}
            #endif

            if(data.size() != rows * columns){throw std::invalid_argument("ste::Matrix::Matrix\n Invalid arguments : sizes must match.");}

           _rows = rows;
           _columns = columns;
           _data = data;

        }

        /***************************************************/
        ///size                                                                                       | Returns the size of the matrix as const std::vector<uint64_t>.
        const std::vector<uint64_t> size() const {return {_rows , _columns};}

        ///columns                                                                                    | Returns the number of columns of the matrix.
        uint64_t columns() const{return _columns;}

        ///rows                                                                                       | Returns the number of rows of the matrix.
        uint64_t rows() const{return _rows; }

        ///lines                                                                                      | Alias for rows().
        uint64_t lines() const{return rows();}

        ///elements                                                                                   | Returns the total number of elements in the matrix.
        uint64_t elements() const{return _rows * _columns;}

        ///clear                                                                                      | Clears all the element in the matrix, and sets its size to (0 ; 0). WARNING : MEMORY IS NOT FREED.
        void clear(){

            _data.clear();
            _rows = 0;
            _columns = 0;

        }

        ///deleteAll                                                                                  | Calls 'delete' on every element, and sets the matrix size to (0 ; 0). Undefined behavior if T is not a pointer.
        void deleteAll(){

            if(_data.size() > 0){

                auto st = _data.begin();
                auto ed = _data.end();

                while (st != ed) {
                    delete *st;
                    ++st;
                }

            }
             clear();
        }

        /***************************************************/

        /// isLine                                                                                    | Returns true if the matrix is a line, false otherwise.
        bool isLine() const{return isRow();}

        /// isRow                                                                                     | Returns true if the matrix is a line, false otherwise.
        bool isRow() const {return (rows() == 1);}

        ///isColumn                                                                                   | Returns true if the matrix is a column, false otherwise.
        bool isColumn() const{return (columns() == 1);}

        ///isSquare                                                                                   | Returns true if the matrix is square, false otherwise.
        bool isSquare() const{return (rows() == columns());}

        ///isInvertible                                                                               | Returns true if the matrix is invertible, false otherwise.
        bool isInvertible() const{return (det() != 0);}

        ///empty                                                                                      | Returns true if the matrix is empty, false otherwise.
        bool empty() const{return _data.empty();}


        /***************************************************/

        ///at                                                                                         | Returns by reference the element at (line , column).
        T& at(const uint64_t &row , const uint64_t &column){

            if(empty()){throw std::out_of_range("ste::Matrix::at\nIndex out of range.");}

            if(row >= rows() || column >= columns()){throw std::out_of_range("ste::Matrix::at\nIndex out of range.");}

            return _data.at(row *columns() + column);

        }

        ///at                                                                                         | Returns by reference the element at the linear index specified.
        T& at(const uint64_t &linear_index){

         if(linear_index >= _data.size()){throw std::out_of_range("ste::Matrix::at\nIndex out of range.");}

         return _data.at(linear_index);

        }

        ///at                                                                                         | Returns the value of the element at (line , column).
        T at(const uint64_t &row , const uint64_t &column) const{

            if(empty()){throw std::out_of_range("ste::Matrix::at\nIndex out of range.");}

            if(row >= rows() || column >= columns()){throw std::out_of_range("ste::Matrix::at\nIndex out of range.");}

            return _data.at(row *columns() + column);

        }

        ///at                                                                                         | Returns the value of the element at the linear index specified.
        T at(const uint64_t &linear_index) const{

         if(linear_index >= rows()*columns()){throw std::out_of_range("ste::Matrix::at\nIndex out of range.");}

         return _data.at(linear_index);


        }

        ///rowAt                                                                                     | Returns the value of the row at the specified index.
        std::vector<T> rowAt(const uint64_t &index) const{

            if(empty()){throw std::out_of_range("ste::Matrix::lineAt\nIndex out of range.");}
            if(index >= rows()){throw std::out_of_range("ste::Matrix::lineAt\nIndex out of range.");}

            std::vector<T> result;
            result.reserve(columns());


            for(uint64_t column = 0 ; column < columns() ; column++){
                result.push_back(at(index , column));
            }
            return result;

        }

        ///lineAt                                                                                    | Returns the value of the row at the specified index.
        std::vector<T> lineAt(const uint64_t &index) const{return rowAt(index);}

        ///columnAt                                                                                  | Returns the value of the column at the specified index.
        virtual std::vector<T> columnAt(const uint64_t &index) const {

            if(index >= columns()){throw std::out_of_range("ste::Matrix::columnAt\nIndex out of range.");}

            std::vector<T> result;
            result.reserve(rows());

            for(uint64_t row = 0 ; row < rows() ; row++){
                result.push_back(at(row , index));
            }

            return result;
        }

                /***************************************************/


        ///replace                                                                                   | Replaces the element at (line , column) by value.
        Matrix& replace(const uint64_t &line , const unsigned &column , const T &value){
            _data.at(line *columns() + column) = value;
            return (*this);
        }

        ///replace                                                                                   | Replaces the element at index (linear index) by value.
        Matrix& replace(const uint64_t &index, const T &value){return replace(index / columns() , index % columns() , value);}

        ///replace                                                                                   | Replaces the line or column specified in argument by value.
        virtual Matrix& replace(const uint64_t &value_index ,const Orientation &orientation , const std::vector<T> &value){


            switch(orientation){

                case(Orientation::LINE):{
                    if(value_index >= rows()){throw std::invalid_argument("ste::Matrix::replace\nCannot replace a line outside the matrix.");}
                    if(value.size() != columns()){throw std::invalid_argument("ste::Matrix::replace\nCannot replace a line by another one with different length.");}


                    for(uint64_t column = 0 ; column < columns() ; column++){
                        replace(value_index , column , value.at(column));
                    }

                    break;
                }
                case(Orientation::COLUMN):{
                    if(value_index  >=  columns()){throw std::invalid_argument("ste::Matrix::replace\nCannot replace a column outside the matrix.");}
                    if(value.size() != rows()){throw std::invalid_argument("ste::Matrix::replace\nCannot replace a column by another one with different length.");}

                        for(uint64_t index_line = 0 ; index_line < rows() ; index_line++){
                            replace(index_line , value_index , value.at(index_line));
                        }

                    break;
                }

                default:{throw std::runtime_error("ste::Matrix::replace\nInvalid orientation provided to add.");}

            }

            return (*this);


        }

        ///replace_row                                                                               | Replaces the line specified in argument by value. WARNING ! If T is dynamically allocated, memory IS NOT freed.
        Matrix& replace_row(const uint64_t &value_index , const std::vector<T> &value){return replace(value_index , Orientation::LINE , value);}

        ///replace_line                                                                              | Replaces the line specified in argument by value. WARNING ! If T is dynamically allocated, memory IS NOT freed.
        Matrix& replace_line(const uint64_t &value_index , const std::vector<T> &value){return replace(value_index , Orientation::LINE , value);}

        ///replace_column                                                                            | Replaces the column specified in argument by value. WARNING ! If T is dynamically allocated, memory IS NOT freed.
        Matrix& replace_column(const uint64_t &value_index , const std::vector<T> &value){return replace(value_index , Orientation::COLUMN , value);}

        /// replace                                                                                  | Replaces the elements in the range specified by value. WARNING ! If T is dynamically allocated, memory IS NOT freed.
        virtual Matrix& replace(const uint64_t &line_begin, const uint64_t &line_end ,
                             const uint64_t &column_begin, const uint64_t &column_end,
                             const T &value){


                if(line_begin < begin_row()     ||
                   line_end > end_row()         ||
                   column_begin < begin_column() ||
                   column_end > end_column()){throw std::invalid_argument("ste::Matrix::replace\nCannot replace an element outside the matrix.");}


                for(uint64_t line = line_begin ; line < line_end ; line++){
                    for(uint64_t column = column_begin ; column < column_end ; column++){
                        _data.at(line *columns() + column) = value;
                    }
                }

                return (*this);

        }


                  /***************************************************/
        ///add                                                                                       | Adds either a column or a line at the end of the matrix containing data according to orientation.
        virtual Matrix& add(const std::vector<T> &data , const Orientation &orientation){

            switch(orientation){

                case(Orientation::LINE):{
                    if(data.size() != columns()){throw std::invalid_argument("ste::Matrix::add\nSizes must match when appending a new line to a matrix.");}
                        _data.insert(_data.end() , data.begin() , data.end());
                    _rows++;

                    break;
                    }
                case(Orientation::COLUMN):{
                    if(data.size() != rows()){throw std::invalid_argument("ste::Matrix::add\nSizes must match when appending a new line to a matrix.");}

                    for(uint64_t row = 0 ; row < rows() ; row++){
                        _data.insert(_data.begin() + ((rows()-row) * (_columns)), data.at(rows()-row-1));
                    }
                    _columns++;

                break;
                }

                default:{throw std::runtime_error("ste::Matrix::add\nInvalid orientation provided to add.");}

            }

            return (*this);

        }

        ///add_row                                                                                   | Adds a row containing data at the end of the matrix.
        Matrix& add_row(const std::vector<T> &data){return add(data , Orientation::LINE);}

        ///add_line                                                                                  | Adds a row containing data at the end of the matrix.
        Matrix& add_line(const std::vector<T> &data){return add(data , Orientation::LINE);}

        ///add_column                                                                                | Adds a column containing data at the end of the matrix.
        Matrix& add_column(const std::vector<T> &data){return add(data , Orientation::COLUMN);}


                  /***************************************************/

        ///push_back                                                                                 | Alias for add.
        Matrix& push_back(const std::vector<T> &data , const Orientation &orientation){return add(data , orientation);}

        ///push_back_row                                                                             | Alias for add_row.
        Matrix& push_back_row(const std::vector<T> &data){return add_line(data);}

        ///push_back_line                                                                            | Alias for add_row.
        Matrix& push_back_line(const std::vector<T> &data){return add_line(data);}

        ///push_back_column                                                                          | Alias for add_colum.
        Matrix& push_back_column(const std::vector<T> &data){return add_column(data);}

                  /***************************************************/

        ///remove                                                                                    | Removes either a column or a line at the specified index containing data according to orientation. WARNING ! If T is dynamically allocated, memory IS NOT freed.
        virtual Matrix& remove(const uint64_t &element_index , const Orientation &orientation){

            switch(orientation){

                case(Orientation::LINE):{

                if(element_index >= _rows){throw std::invalid_argument("ste::Matrix::remove\nCannot remove a line outside the matrix.");}

                _data.erase(_data.begin() + element_index * _columns , _data.begin() + element_index*_columns + _columns);

                _rows--;
                _columns = (_rows > 0) ? _columns : 0; //If the matrix only contained one row, it is now empty.

                break;
                }
                case(Orientation::COLUMN):{

                if(element_index >= _columns){throw std::invalid_argument("ste::Matrix::remove\nCannot remove a column outside the matrix.");}

                for(uint64_t row = 0 ; row < _rows ; row++){
                _data.erase(_data.begin() + (_rows-row-1) * _columns + element_index);
                }
                _columns--;
                _rows = (_columns > 0) ? _rows : 0; //If the matrix only contained one column, it is now empty.

                break;
                }

                default:{throw std::runtime_error("ste::Matrix::remove\nInvalid orientation provided to remove.");}

            }

            return (*this);
        }

        ///remove_row                                                                                | Removes a row at the specified index. WARNING ! If T is dynamically allocated, memory IS NOT freed.
        Matrix& remove_row(const uint64_t &index){return remove(index, Orientation::LINE);}

        ///remove_line                                                                               | Removes a row at the specified index. WARNING ! If T is dynamically allocated, memory IS NOT freed.
        Matrix& remove_line(const uint64_t &index){return remove(index, Orientation::LINE);}

        ///remove_column                                                                             | Removes a column at the specified index. WARNING ! If T is dynamically allocated, memory IS NOT freed.
        Matrix& remove_column(const uint64_t &index){return remove(index, Orientation::COLUMN);}


                  /***************************************************/


        ///insert                                                                                    | Inserts a column or a line containing data at the specified index.
        virtual Matrix& insert(const uint64_t &element_index , const Orientation &orientation , const std::vector<T> &data){

            switch(orientation){

                case(Orientation::LINE):{
                    if(data.size() != columns()){throw std::invalid_argument("ste::Matrix::insert\nCannot insert a line which does not have the same length as the others.");}
                    if(element_index>rows()){return insert(rows() , orientation , data);}

                    _data.insert(_data.begin() + element_index * _columns , data.begin() , data.end());

                    _rows++;


                break;
                }
                case(Orientation::COLUMN):{
                    if(data.size() != rows()){throw std::invalid_argument("ste::Matrix::insert\nCannot insert a column which does not have the same length as the others.");}
                    if(element_index>columns()){return insert(columns() , orientation , data);}

                    ///TODO
                    for(uint64_t row = 0 ; row < rows() ; row++){
                    _data.insert(_data.begin() + (_rows-row-1) * _columns + element_index , data.at(_rows-row-1));
                    }
                    _columns++;

                break;
                }

                default:{throw std::runtime_error("ste::Matrix::insert\nInvalid orientation provided to insert.");}

            }

            return (*this);

        }

        ///insert_row                                                                                | Inserts a row at index containing data.
        Matrix& insert_row(const uint64_t &index , const std::vector<T> &data){return insert(index, Orientation::LINE , data);}

        ///insert_line                                                                               | Inserts a row at index containing data.
        Matrix& insert_line(const uint64_t &index , const std::vector<T> &data){return insert(index, Orientation::LINE , data);}

        ///insert_column                                                                             | Inserts a column at index containing data.
        Matrix& insert_column(const uint64_t &index , const std::vector<T> &data){return insert(index, Orientation::COLUMN , data);}

        /***************************************************/

        ///swap                                                                                      | Swaps two rows or two columns at the positions specified.
        virtual Matrix& swap(const uint64_t &element_1 , const uint64_t element_2 ,const Orientation &orientation){

            switch(orientation){

                case(Orientation::LINE):{

                    if(element_1 >= rows() || element_2 >= rows()){throw std::invalid_argument("ste::Matrix::swap\nCannot swap lines outside the matrix.");}

                    std::vector<T> temp = rowAt(element_1);
                    replace_line(element_1 , rowAt(element_2));
                    replace_line(element_2 , temp);

                break;
                }
                case(Orientation::COLUMN):{

                    if(element_1 >= columns() || element_2 >= columns()){throw std::invalid_argument("ste::Matrix::swap\nCannot swap columns outside the matrix.");}

                    std::vector<T> temp = columnAt(element_1);

                    replace_column(element_1 , columnAt(element_2));
                    replace_column(element_2 , temp);


                break;
                }

                default:{throw std::runtime_error("ste::Matrix::swap\nInvalid orientation provided to insert.");}

            }

            return (*this);

        }

        ///swap_rows                                                                                 | Convenience function to swap two rows at a specified positions.
        Matrix& swap_rows(const uint64_t &element_1 , const uint64_t element_2){return swap(element_1, element_2 ,Orientation::LINE);}

        ///swap_lines                                                                                | Convenience function to swap two rows at a specified positions.
        Matrix& swap_lines(const uint64_t &element_1 , const uint64_t element_2){return swap(element_1, element_2 ,Orientation::LINE);}

        ///swap_columns                                                                              | Convenience function to swap two columns at a specified positions.
        Matrix& swap_columns(const uint64_t &element_1 , const uint64_t element_2){return swap(element_1, element_2 ,Orientation::COLUMN);}

        /***************************************************/

        Matrix& reshape(const uint64_t &rows , const uint64_t &columns){

            if((rows * columns) != elements() ){throw std::invalid_argument("ste::Matrix::reshape\nInvalid size. Cannot change the total number of elements in the matrix while reshaping it.");}

            _rows = rows;
            _columns = columns;

            return (*this);

        }

        /***************************************************/

        ///toVector2D                                                                                | Converts the matrix to std::vector<std::vector<T>>.
        virtual std::vector<std::vector<T>> toVector2D() const{

            std::vector<std::vector<T>> result;
            result.reserve(rows());

            for(uint64_t row = 0 ; row < rows() ; row++){
                std::vector<T> line;
                line.reserve(columns());

                for(uint64_t column = 0 ; column < columns() ; column++){
                    line.push_back(at(row , column));
                }

                result.push_back(line);
            }

            return result;

        }

        ///toVector1D                                                                                | Converts the matrix to std::vector<T>.
        virtual const std::vector<T> toVector1D() const{return _data;}

        ///toVector1D                                                                                | Converts the matrix to std::vector<T>.
        virtual std::vector<T> toVector1D(){return _data;}


        /***************************************************/

        ///print                                                                                     | Prints the contents of the matrix in stdout.
        virtual void print() const{

            std::cout << "[ ";
            for(uint64_t row = 0 ; row < rows() ; row++){
            std::cout << "[ ";
                for(uint64_t column = 0 ; column < columns() ; column++){
                    std::cout << at(row , column) << " ";
                }
            std::cout << "] ";
            }

            std::cout << "]" << std::endl; //Buffer flushed to prevent user misusage

        }

        ///print_size                                                                                | Prints the size of the matrix in stdout.
        virtual void print_size() const{
            std::cout << "[" << rows() << " ; "<< columns() << "]" << std::endl; //Buffer flushed to prevent user misusage
        }

        /***************************************************/

        ///begin_row                                                                                 | Convenience function that returns the beginning of a row. Provided to obtain syntax as close as std::algorithm functions as possible.
        virtual uint64_t begin_row() const{return 0;}
        ///begin_line                                                                                | Convenience function that returns the beginning of a row. Provided to obtain syntax as close as std::algorithm functions as possible.
        uint64_t begin_line() const{return begin_row();}
        ///begin_column                                                                              | Convenience function that returns the beginning of a column. Provided to obtain syntax as close as std::algorithm functions as possible.
        virtual uint64_t begin_column() const{return 0;}

        ///end_row                                                                                   | Convenience function that returns the end of a row. Provided to obtain syntax as close as std::algorithm functions as possible.
        virtual uint64_t end_row() const{return rows();}
        ///end_line                                                                                  | Convenience function that returns the end of a row. Provided to obtain syntax as close as std::algorithm functions as possible.
        uint64_t end_line() const{return end_row();}
        ///end_column                                                                                | Convenience function that returns the end of a column. Provided to obtain syntax as close as std::algorithm functions as possible.
        virtual uint64_t end_column() const{return columns();}

                /***************************************************/

        ///for_each                                                                                 | Analog to std::for_each. Applies the function in argument to every element in the matrix.
        void for_each(std::function<T(T&)> function){
            std::for_each(_data.begin() , _data.end() , function);
        }

        ///for_each                                                                                 | Analog to std::for_each. Applies the function in argument to every element in the matrix.
        void for_each(std::function<void(T&)> function){
            std::for_each(_data.begin() , _data.end() , function);
        }

        ///for_each                                                                                 | Analog to std::for_each. Applies the function in argument to every element in the matrix.
        void for_each(std::function<void(const T&)> function) const{
            std::for_each(_data.begin() , _data.end() , function);
        }


        ///for_each                                                                                 | Analog to std::for_each. Applies the function in argument to every element in the matrix.
        static void for_each(Matrix &matrix , std::function<T(T&)> function){
            matrix.for_each(function);
        }

        ///for_each                                                                                 | Analog to std::for_each. Applies the function in argument to every element in the matrix.
        static void for_each(Matrix &matrix , std::function<void(T&)> function){
            matrix.for_each(function);
        }


        ///for_each                                                                                 | Analog to std::for_each. Applies the function in argument to every element in the matrix.
        static void for_each(const Matrix &matrix , std::function<void(const T&)> function){
            matrix.for_each(function);
        }




        ///transform                                                                                | Analog to std::transform. Applies the function in argument to every element in the matrix and modifies them according to its return value.
        void transform(std::function<T(T&)> function){
            std::transform(_data.begin() , _data.end() , _data.begin() , function);
        }

        ///transform                                                                                | Analog to std::transform. Applies the function in argument to every element in the matrix and modifies them according to its return value.
        static void transform(Matrix &matrix , std::function<T(T&)> function){
            matrix.transform(function);
        }

                /***************************************************/

        ///sum                                                                                        | Returns the sum of all elements of the matrix, as T (meaning that overflow may occur).
        virtual T sum() const{
            T accumulator = T(0);

            for(uint64_t index_line = 0 ; index_line < rows() ; index_line++){
                for(uint64_t index_column = 0 ; index_column < columns() ; index_column++){
                    accumulator = accumulator + at(index_line , index_column); //+= is not used as it is not necesseraly provided for all T (especially when T is a class).
                }
            }

            return accumulator;
        }

        ///mean                                                                                       | Returns the mean value of all elements of the matrix, as T (meaning that rounding error and overflow may occur). It is computed as sum()/(size.at(0)*size.at(1))
        virtual T mean() const{return average();}

        ///average                                                                                    | Convenience function that returns mean().
        virtual T average() const{return sum() / (rows()*columns());}

        ///max                                                                                        | Returns the value of the maximum element (according to std::max_element) in the matrix.
        virtual T max() const{return *std::max_element(_data.begin() , _data.end());}

        ///min                                                                                        | Returns the value of the minimum element (according to std::min_element) in the matrix.
        virtual T min() const{return *std::min_element(_data.begin() , _data.end());}


                /***************************************************/

        ///trace                                                                                      | Returns the trace of the matrix, computed as T (meaning that rounding error and overflow may occur). Throws an exception (std::invalid_argument) if the matrix is not square.
        virtual T trace() const{

            //Not enough calculations to justify GPU here
            if(!isSquare()){throw std::invalid_argument("ste::Matrix::trace\nMatrix is not square.");}

            T accumulator = T(0);

            for(uint64_t index = 0 ; index < rows() ; index++){
                accumulator += T(at(index , index));
            }

            return accumulator;


        }


        ///det                                                                                        | Returns the determinant of the matrix. Throws an exception (std::invalid_argument) is the matrix is not square.
        virtual T det() const{

            if(!isSquare()){throw std::invalid_argument("ste::Matrix::det\nMatrix is not square.");}

            if(rows() == 1 && columns() == 1){return at(0 , 0);}
            if(rows() == 2 && columns() == 2){return at(0 , 0) * at(1 , 1) - at(0 , 1)*at(1 , 0);}


            //#ifdef USE_GPU
            //return T(CUDA_det(toVector1D() , rows() , columns()));
            //#else
            T determinant = T(0);

            for(uint64_t element = 0 ; element < rows() ; element++){

                determinant = determinant +
                                            (at(0 , element) *
                                            std::pow(-1 , element) *
                                            Matrix((*this)).remove_row(0).remove_column(element).det());
            }


            return determinant;
            //#endif



        }

        ///cofactor                                                                                   | Returns the cofactor of the specified line and column.
        virtual T cofactor(uint64_t line , uint64_t column) const{

            if(line >= rows()){throw std::invalid_argument("ste::Matrix::cofactor\nLine is outside the matrix.");}
            if(column >= columns()){throw std::invalid_argument("ste::Matrix::cofactor\nColumn is outside the matrix.");}

            Matrix temp(_data , rows() , columns());
            temp.remove_line(line);
            temp.remove_column(column);

            return std::pow(T(-1) , line+column) * temp.det();


        }


        ///cofactor                                                                                   | Returns the cofactor of the specified linear index.
        virtual T cofactor(uint64_t index) const{return cofactor(index / columns() , index % columns());}


        ///comatrix                                                                                   | Returns the cofactor matrix. Convenience function that returns cofactormatrix().
        virtual Matrix comatrix() const {return cofactormatrix();}


        ///cofactormatrix                                                                             | Returns the cofactor matrix.
        virtual Matrix cofactormatrix() const {


            //#ifdef USE_GPU
            //return Matrix(CUDA_cofactormatrix(toVector1D() , rows() , columns()) , rows() , columns());
            //#else
            Matrix result(_data , rows() , columns());

            for(uint64_t index = 0 ; index < rows()*columns() ; index++){

                Matrix temp(_data , rows() , columns());

                T value = temp.cofactor(index);
                result.replace(index , value);

            }


            return result;
            //#endif
        }


        ///transpose                                                                                  | Returns the transpose of the matrix.
        virtual Matrix transpose() const{


        //#ifdef USE_GPU
        //return Matrix(CUDA_transpose(toVector1D(), rows() , columns()) , rows() , columns());
        //#else

            Matrix result(columns() , rows());

            for(uint64_t row = 0 ; row < rows() ; row++){
                for(uint64_t column = 0 ; column < columns() ; column++){
                    result.replace(column , row , at(row  , column));
                }
            }
            return result;

       // #endif
        }



        ///inv                                                                                        | Returns the inverse of the matrix as computed by operator!.
        Matrix inv(){return !(*this);}

        ///inverse                                                                                    | Returns the inverse of the matrix as computed by operator!.
        Matrix inverse(){return inv();}

        ///invert                                                                                     | Returns the inverse of the matrix as computed by operator!.
        static Matrix invert(const Matrix &arg){return !arg;}

        /***************************************************/

        ///hadamard                                                                                    | Returns the Hadamard product of two matrices. Throws an exception if the sizes do not match.
        virtual Matrix hadamard(const Matrix &arg) const{

            if(size() != arg.size()){throw std::invalid_argument("Matrix::hadamard\nSizes of the two matrices must match."
                                                                 + std::string("First argument length: [") + std::to_string(rows()) + " ; "+ std::to_string(columns()) + "]\n"
                                                                 + std::string("Second argument length: [") + std::to_string(arg.rows()) + " ; "+ std::to_string(arg.columns()) + "]\n");}

           //#ifdef USE_GPU
           /*return Matrix(CUDA_hadamard(
                             toArray()     , rows()     , columns(),
                             arg.toArray() , arg.rows() , arg.columns()
                             ) ,
                         rows() , columns());*/
           //#else


            Matrix result(_data , _rows , _columns);


            for(uint64_t item = 0 ; item < rows() * columns() ; item++){
                result.at(item) = at(item) * arg.at(item);
            }

            return result;
           //#endif




        }

        ///hadamard                                                                                    | Returns the Hadamard product of two matrices. Throws an exception if the sizes do not match.
        static Matrix hadamard(const Matrix &arg1 , const Matrix &arg2){return arg1.hadamard(arg2);}

        ///element_wise                                                                                | Convenience function that returns the Hadamard product of two matrices.
        Matrix element_wise(const Matrix &arg){return hadamard(arg);}

        ///element_wise                                                                                | Convenience function that returns the Hadamard product of two matrices.
        static Matrix element_wise(const Matrix &arg1 , const Matrix &arg2){return hadamard(arg1 , arg2);}

                /***************************************************/

        ///fill                                                                                        | Resizes the matrix to [size ; size] and fills it with value. WARNING ! Any dynamically allocated content IS NOT deleted when this function is called.
        virtual void fill(const uint64_t &size , const T &value){
            fill(size , size , value);
        }

        ///fill                                                                                        | Resizes the matrix to [length ; width] and fills it with value. WARNING ! Any dynamically allocated content IS NOT deleted when this function is called.
        virtual void fill(const uint64_t &length , const uint64_t &width , const T &value){
            _data = std::vector<T> (length * width , value);
        }


                  /***************************************************/

        ///zeroes                                                                                      | Returns a matrix of dimension [size ; size] and filled with T(0).
        static Matrix zeroes(const uint64_t &size){
            return zeroes(size , size);
        }

        ///zeroes                                                                                      | Returns a matrix of dimension [rows ; columns] and filled with T(0).
        static Matrix zeroes(const uint64_t &rows , const uint64_t &columns){
            Matrix result(rows , columns);
            result.fill(rows , columns , T(0));
            return result;
        }


                    /***************************************************/
        ///ones                                                                                        | Returns a matrix of dimension [size ; size] and filled with T(1).
        static Matrix ones(const uint64_t &size){
            return ones(size , size);
        }

        ///ones                                                                                        | Returns a matrix of dimension [rows ; columns] and filled with T(1).
        static Matrix ones(const uint64_t &rows , const uint64_t &columns){
            Matrix result(rows , columns);
            result.fill(rows , columns , T(1));
            return result;
        }

                 /***************************************************/

        ///eye                                                                                         | Returns the identity matrix of the specified size.
        static Matrix eye(const uint64_t &size){
            Matrix result = zeroes(size);

            for(uint64_t line = 0 ; line < size ; line++){
                result.replace(line , line , 1);
            }

            return result;

        }



                /***************************************************/


        static Matrix randn(const uint64_t &rows , const uint64_t &columns , const T&mean = T(0) , const T&standard_deviation = T(1)){

           std::mt19937 random_device(time(0));
           std::normal_distribution<T> initializer(mean, standard_deviation);

           auto gen = std::bind(initializer , random_device);

           std::vector<T> elements;
           elements.resize(rows *columns);

           std::generate(elements.begin() , elements.end() , gen);

           return Matrix(elements , rows , columns);


        }



        static Matrix randn(const uint64_t &size , const T&mean = T(0) , const T&standard_deviation = T(1.)){
            return randn(size , size , mean , standard_deviation);
        }


                /***************************************************/


         static Matrix uniform(const uint64_t &rows , const uint64_t &columns , const T&min = T(0) , const T&max = T(1)){

             std::mt19937 random_device(time(0));
             std::uniform_real_distribution<T> initializer(min, max);

             auto gen = std::bind(initializer , random_device);

             std::vector<T> elements;
             elements.resize(rows *columns);

             std::generate(elements.begin() , elements.end() , gen);

             return Matrix(elements , rows , columns);


         }

         static Matrix uniform(const uint64_t &size , const T&min = T(0) , const T&max = T(1)){
            return uniform(size , size , min , max);
         }


         static Matrix uniform_int(const uint64_t &rows , const uint64_t &columns , const T&min = T(0) , const T&max = T(1)){

             std::mt19937 random_device(time(0));
             std::uniform_int_distribution<T> initializer(min, max);

             auto gen = std::bind(initializer , random_device);

             std::vector<T> elements;
             elements.resize(rows *columns);

             std::generate(elements.begin() , elements.end() , gen);

             return Matrix(elements , rows , columns);


         }

         static Matrix uniform_int(const uint64_t &size , const T&min = T(0) , const T&max = T(1)){
            return uniform(size , size , min , max);
         }

                /***************************************************/

        static Matrix rand(const uint64_t &rows , const uint64_t &columns , const T&min = T(0) , const T&max = T(1)){
            return uniform(rows , columns , min , max);
        }

        static Matrix rand(const uint64_t &size , const T&min = T(0) , const T&max = T(1)){
           return uniform(size , size , min , max);
        }

                /***************************************************/



        ///operator=                                                                                 | Assignement operator.
        virtual void operator=(const Matrix &arg){
            _data = arg._data;
            _rows = arg._rows;
            _columns = arg._columns;
        }

        ///operator=                                                                                 | Assignement operator. Changes the data to the one from arg if possible, throws std::invalid_argument otherwise.
        virtual void operator=(const std::vector<std::vector<T>> &arg){

            if(arg.empty()){throw std::invalid_argument("ste::Matrix::operator=\nCannot construct a matrix from empty data.");}

            const uint64_t column_length = arg.at(0).size();

            for(const std::vector<T> &line:arg){
                if(line.size() != column_length){throw std::invalid_argument("ste::Matrix::operator=\nCannot construct a matrix with irregular column size.");}
            }

            _rows = arg.size();
            _columns = column_length;

            _data.clear();
            _data.reserve(rows() * columns());

            for(auto &line:arg){
                _data.insert(_data.end() , line.begin() , line.end());
            }

        }


                                                            /*-------------------*/


        ///operator*                                                                                 | Multiplies two matrices using the usual matrix product definition.
        virtual Matrix operator* (const Matrix &arg) const{

            if(columns() != arg.rows()){throw std::invalid_argument("ste::Matrix::operator*\nDimension mismatch.\nFirst argument size: [ "
                                                                    + std::to_string(rows()) + " ; " + std::to_string(columns()) + "]\n"
                                                                    + "Second argument size: [ " + std::to_string(arg.rows()) + " ; " + std::to_string(arg.columns()) +"].") ;}

            #ifdef USE_GPU

            return Matrix(
                          CUDA_mult_MAT(toVector1D() , rows() , columns () , arg.toVector1D() , arg.rows() , arg.columns()) ,
                          rows() ,
                          arg.columns());

            #else

            Matrix result(rows() , arg.columns());

            for(uint64_t index_line = 0 ; index_line < rows() ; index_line++){

                for(uint64_t index_column = 0 ; index_column < arg.columns() ; index_column++){
                    T value = T(0);
                    for(uint64_t index_sum = 0 ; index_sum < columns(); index_sum++){
                        value = value + T(at(index_line,index_sum)) * T(arg.at(index_sum,index_column)); //Avoid use of operator +=, due to some classes not having it implemented.
                    }
                    result.replace(index_line , index_column , value);
                }

            }


            return result;

            #endif


        }

        ///operator*=                                                                                | Multiplies the matrix by the argument using the usual matrix product definition, and returns a reference to it.
        virtual Matrix& operator*=(const Matrix &arg){

            #ifdef USE_GPU
            (*this) = Matrix(
                   CUDA_mult_MAT(toVector1D() , rows() , columns () , arg.toVector1D() , arg.rows() , arg.columns()) ,
                   rows() ,
                   arg.columns());

            #else




            const Matrix old_this = (*this);
            reshape(rows() , arg.columns());

            for(uint64_t row = 0 ; row < old_this.rows() ; row++){

                for(uint64_t column = 0 ; column < arg.columns() ; column++){
                    T value = T(0);
                    for(uint64_t sum = 0 ; sum < columns(); sum++){
                        value = value + T(old_this.at(row,sum)) * T(arg.at(sum,column)); //Avoid use of operator +=, due to some classes not having it implemented.
                    }
                    replace(row , column , value);
                }

            }
            #endif
            return (*this);
        }


        ///operator*                                                                                 | Multiplies all elements of the matrix by arg.
        virtual Matrix operator* (const T &arg) const{

            Matrix result(_data , rows() , columns());

            if(arg == 1){return result;}

            std::transform(result._data.begin() , result._data.end() , result._data.begin() ,
                           std::bind(std::multiplies<T>() , std::placeholders::_1 , arg));

            return result;

        }

        ///operator*=                                                                                | Multiplies all elements of the matrix by arg,x, and returns a reference to it.
        virtual Matrix& operator*=(const T &arg){

            if(arg == 1){return (*this);}

            std::transform(_data.begin() , _data.end() , _data.begin() ,
                           std::bind(std::multiplies<T>() , std::placeholders::_1 , arg));

            return (*this);

        }


                                                    /*-------------------*/


        ///operator+                                                                                 | Adds T two matrices.
        virtual Matrix operator+ (const Matrix &arg) const{

           Matrix result(_data , rows() , columns());
           std::transform(result._data.begin(), result._data.end(), arg._data.begin(), result._data.begin(), std::plus<T>());
           return result;

        }

        ///operator+=                                                                                | Adds arg to all the elements of the Matrix, and returns a reference to it.
        virtual Matrix& operator+=(const Matrix &arg){

            std::transform(_data.begin(), _data.end(), arg._data.begin(), _data.begin(), std::plus<T>());

            return (*this);
        }


        ///operator+                                                                                 | Adds arg to all elements. May be overrided for other purposes.
        virtual Matrix operator+ (const T &arg) const{

            Matrix result(_data , rows() , columns());
            std::for_each(result._data.begin(), result._data.end(), [arg](T& value) { value = value + arg;}); //Avoid use of operator+=, due to some classes not having it implemented.
            return result;

        }

        ///operator+=                                                                                | Adds arg to all the elements of the Matrix, and returns a reference to it.
        virtual Matrix& operator+=(const T &arg){
            std::for_each(_data.begin(), _data.end(), [arg](T& value) { value = value + arg;}); //Avoid use of operator+=, due to some classes not having it implemented.
            return (*this);
        }


                                                    /*-------------------*/


        ///operator-                                                                                 | Substracts two matrices.
        virtual Matrix operator- (const Matrix &arg) const {

            Matrix result(_data , rows() , columns());
            std::transform(result._data.begin(), result._data.end(), arg._data.begin(), result._data.begin(), std::minus<T>());
            return result;

        }

        ///operator-=                                                                                | Adds arg to all the elements of the Matrix, and returns a reference to it.
        virtual Matrix& operator-=(const Matrix &arg){

            std::transform(_data.begin(), _data.end(), arg._data.begin(), _data.begin(), std::minus<T>());

            return (*this);
        }

        ///operator-                                                                                 | Substracts arg to all elements of the matrix. May be overrided for other purposes.
        virtual Matrix operator- (const T &arg) const {

            Matrix result(_data , rows() , columns());
            std::for_each(result._data.begin(), result._data.end(), [arg](T& value) { value = value - arg;}); //Avoid use of operator+-, due to some classes not having it implemented.
            return result;
        }

        ///operator-=                                                                                | Substracts arg to all the elements of the Matrix, and returns a reference to it.
        virtual Matrix& operator-=(const T &arg){
            std::for_each(_data.begin(), _data.end(), [arg](T& value) { value = value - arg;}); //Avoid use of operator+=, due to some classes not having it implemented.
            return (*this);
        }

                                                            /*-------------------*/



        ///operator!                                                                                 | Returns the inverse of the matrix, or an empty one if not inversible.
        Matrix operator! () const{                   //Inverse of matrix

            const T determinant = det();
            if(determinant == 0){
                std::cerr << "ste::Matrix::operator! : WARNING ! Matrix is not inversible. Returned an empty matrix.\n";
                return Matrix();
            }

            return Matrix(_data , rows() , columns()).cofactormatrix().transpose()*(1/determinant);

        }

                                                            /*-------------------*/


        ///operator^                                                                                 | Returns the matrix to the specified power (usual matrix product is used).
        virtual Matrix operator^ (const long long int &arg) const{ //Power operator

            if(arg == 0){

                if(!isSquare()){throw std::invalid_argument("ste::Matrix::operator^\n Cannot use power 0 for non-square matrices.");}

                return eye(rows());
            }

            Matrix output(_data , rows() , columns());

            if(arg < 0){output = !output;}

            for(long long int power = 1 ; power < std::abs(arg) ; power++){output = output * (*this);}

            return output;
        }


        ///operator^                                                                                 | Raises the matrix to the specified power (usual matrix product is used), and returns a reference to it.
        virtual Matrix& operator^=(const long long int &arg){

            if(arg == 0){

                if(!isSquare()){throw std::invalid_argument("ste::Matrix::operator^\n Cannot use power 0 for non-square matrices.");}
                (*this) = eye(rows());

                return (*this);
            }

            if(arg < 0){(*this) = !(*this);}


            for(long long int power = 1 ; power <  std::abs(arg) ; power++){(*this) = (*this) * (*this);}

            return (*this);

        }

                                                            /*-------------------*/

        ///operator==                                                                               | Equality operator. Returns true only if all elements are identical and at the same position.
        virtual bool operator== (const Matrix &arg) const{return ((_data == arg._data) && (_rows == arg._rows) && (_columns == arg._columns));}

        ///operator!=                                                                               | Returns the opposite of the result given by operator==.
        virtual bool operator!= (const Matrix &arg) const{return !(*this == arg);}




};//class Matrix


/** Convenience TYPES */

typedef Matrix<float> FMatrix ;
typedef Matrix<double> DMatrix;
typedef Matrix<long double> LDMatrix;

typedef Matrix<int> IMatrix;
typedef Matrix<long int> LIMatrix;
typedef Matrix<long long int> LLIMatrix;

typedef Matrix<unsigned int> UIMatrix;
typedef Matrix<unsigned long> ULMatrix;
typedef Matrix<unsigned long long> ULLMatrix;

typedef Matrix<char> CMatrix;
typedef Matrix<unsigned char> UCMatrix;


} //namespace ste
#endif // MATRIX_H
