# ste::Matrix

C++ class that provides an interface for matrix-based computing.


**FOR DETAILED DOCUMENTATION, SEE /documentation**


# Features

 • Can hold any class.
 
 • Possibility to use GPU for calculations. **\[WIP\]** : currently, only multiplications and transpose are available.
 
 • Fast conversion to `std::vector<T>` to facilitate GPU-acceleration-based algorithms. 

 

 • Operators `*`, `+` and `-` available as long as the template parameters provides these operators.

 

 • Determinant, inverse, tranpose ,cofactormatrix, trace, mean, average. 
 
 • Classic `fill`, `zeroes`, `ones`, `eye`, `randn` and `rand`. 
 
 • Dynamic resizing (possibility to add, remove and inverst lines and / or columns). 
 
 • Fast reshaping (`O(1)`).

 • Possibility to directly print contents and size to `stdout` or any `std::ostream`.
 
 • Possibility to override most functions in subclasses to increase performances in specific cases.


## Convenience types

|                |Type                           |Shortcut                   |
|----------------|-------------------------------|---------------------------|
|                |`Matrix<float>`                |`FMatrix`                  |
|                |`Matrix<double>`               |`DMatrix`                  |
|                |`Matrix<long double>`          |`LDMatrix`                 |
|                |`Matrix<int>`                  |`IMatrix`                  |
|                |`Matrix<long int>`             |`LIMatrix`                 |
|                |`Matrix<long long int>`        |`LLIMatrix`                |
|                |`Matrix<unsigned int>`         |`UIMatrix`                 |
|                |`Matrix<unsigned long>`        |`ULMatrix`                 |
|                |`Matrix<unsigned long long>`   |`ULLMatrix`                |
|                |`Matrix<char>`                 |`CMatrix`                  |
|                |`Matrix<unsigned char>`        |`UCMatrix`                 |

# Enum

## Member enums
 
|Name                                    |Contents        |Description                                                              |Notes                                                                                                                         |
|----------------------------------------|----------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
|`enum class ste::Matrix<>::Orientation` |`ROW` , `COLUMN`| Used to specify if a function needs to be applied to a row or a column. |This enum holds redundant members : `LINE`, `RW`, `R` are equivalent to `ROW`. `COL` , `CL` , `C` are equivalent to `COLUMN`. |

## Non-member enum

|Name                  |Contents     |Description                                         |Notes                                                                                                                                                                |
|----------------------|-------------|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`enum class ste::EXE` |`CPU` , `GPU`| Used to specify the execution policy for a matrix. | See Paragraph `Using the GPU` for more details. This enum holds redundant members :, `C`, `HOST` are equivalent to `CPU`. `G` , `DEVICE` , are equivalent to `GPU`. |

**N.B** : `ste::EXE::GPU` and its alias are only available if `STE_MATRIX_ALLOW_GPU` is `#define`d.


# Member functions
> Virtual functions are marqued _**[v]**_.

> Static functions are marqued **[S]**.


### Constructor:
|      |Function  |Description                                                                                                                                                 |
|------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      | `Matrix` | Constructor. Can accept a size (rows , columns or same for both) , a `std::vector<std::vector<T>>` or a `std::vector<T>` and a size to construct a Matrix. |

 
### Accessors:

|      |Function     |Description                                                                    |Notes                                                |
|------|-------------|-------------------------------------------------------------------------------|-----------------------------------------------------|
|      |`size`       | Returns the size of the matrix, as `const std::vector<uint64_t>`.             |                                                     |
|      |`columns`    | Returns the number of columns of the matrix.                                  |                                                     |
|      |`rows`       | Returns the number of rows of the matrix.                                     |                                                     |
|      |`lines`      | Alias for `'rows'`.                                                           |                                                     |
|      |`device`     | Returns the device on which the operations involving the matrix will be made. |                                                     |
|      |`elements`   | Returns the total number of elements in the matrix.                           |                                                     |
|      |`clear`      | Clears all the element in the matrix, and sets its size to (0 ; 0).           | **WARNING : MEMORY IS NOT FREED.**                  |
|      |`delete_all` | Calls `'delete'` on every element, and sets the matrix size to (0 ; 0).       | **Only available when T is dynamically allocated.** |


 
### Information about the matrix shape:

|      |Function        |Description                                                 |
|------|----------------|------------------------------------------------------------|
|      |`is_row`        | Returns true if the matrix is row, false otherwise.        |
|      |`is_line`       | Alias for `'is_row'`.                                       |
|      |`is_column`     | Returns true if the matrix is a column, false otherwise.   |
|      |`is_square`     | Returns true if the matrix is square, false otherwise.     |
|      |`is_invertible` | Returns true if the matrix is invertible, false otherwise. |
|      |`empty`          | Returns true if the matrix is empty, false otherwise.      |

 
### Access to the matrix' contents:

|      |Function    |Description                                                                                                                                  |
|------|------------|---------------------------------------------------------------------------------------------------------------------------------------------|
|      |`at`        | Returns the element at the index specified in argument. It is passed by reference when the matrix is non-const. Linear indexes can be used. |
|      |`row_at`    | Returns the row at the specified index. It is passed by reference when the matrix is non-const.                                             |
|      |`line_at`   | Alias for `'row_at'`.                                                                                                                        |
|      |`column_at` | Returns the column at the specified index. It is always passed by value.                                                                    |

 
### Replacement:

|      |Function             |Description                                                                                       |Notes                                                                |
|------|---------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
|      |`replace`            | Replaces the element, the row or the column specified in argument by the value in last argument. | **WARNING ! If `T` is dynamically allocated, memory IS NOT freed.** |
|      |`replace_row`        | Replaces a row by the one specified in argument.                                                 | **WARNING ! If `T` is dynamically allocated, memory IS NOT freed.** |
|      |`replace_line`       | Alias for `'replace_row'`.                                                                       | **WARNING ! If `T` is dynamically allocated, memory IS NOT freed.** |
|      |`replace_column`     | Replaces a column by the one specified in argument.                                              | **WARNING ! If `T` is dynamically allocated, memory IS NOT freed.** |



### Appending to the matrix:
|      |Function           |Description                                                             |
|------|-------------------|------------------------------------------------------------------------|
|      |`add`              | Adds either a line or a column from a vector at the end of the matrix. |
|      |`add_row`          | Convenience function to add a row at the end of the matrix.            |
|      |`add_line`         | Alias for `'add_row'`.                                                 |
|      |`add_column`       | Convenience function to add a column at the end of the matrix.         |
|      |`push_back`        | Alias for `'add'`.                                                     |
|      |`push_back_row`    | Alias for `'add_row'`.                                                 |
|      |`push_back_line`   | Alias for `'add_line'`.                                                |
|      |`push_back_column` | Alias for `'add_column'`.                                              |

 
### Removing from the matrix:
|      |Function           |Description                                                       | Notes                                                             |
|------|-------------------|------------------------------------------------------------------|-------------------------------------------------------------------|
|      |`remove`           | Removes either a line or a column at the position specified.     | **WARNING ! If T is dynamically allocated, memory IS NOT freed.** |
|      |`remove_row`       | Convenience function to remove a row at a specified position.    | **WARNING ! If T is dynamically allocated, memory IS NOT freed.** |
|      |`remove_line`      | Alias for `'remove_row'.`                                        | **WARNING ! If T is dynamically allocated, memory IS NOT freed.** |
|      |`remove_column`    | Convenience function to remove a column at a specified position. | **WARNING ! If T is dynamically allocated, memory IS NOT freed.** |
|      |`cut`              |  Removes the rows or columns specified.                          | **WARNING ! If T is dynamically allocated, memory IS NOT freed.** |
|      |`cut_rows`         | Convenience function to remove the specified rows.               | **WARNING ! If T is dynamically allocated, memory IS NOT freed.** |
|      |`cut_lines`        | Alias for `'cut_rows'`.                                          | **WARNING ! If T is dynamically allocated, memory IS NOT freed.** |
|      |`cut_columns`      | Convenience function to remove the specified columns.            | **WARNING ! If T is dynamically allocated, memory IS NOT freed.** |
 
### Insertion:
|      |Function           |Description                                                             |
|------|-------------------|------------------------------------------------------------------------|
|      |`insert`           | Inserts either a line or a column at the position specified.           |
|      |`insert_row`       | Convenience function to insert a line at a specified position.         |
|      |`insert_line`      | Alias for `'insert_row'`.                                              |
|      |`insert column`    | Convenience function to insert a column at a specified position.       |

 
### Swaping elements:
|      |Function           |Description                                                             |
|------|-------------------|------------------------------------------------------------------------|
|      |`swap`             | Swaps two lines or two columns at the positions specified.             |
|      |`swap_row`         | Convenience function to swap two rows at a specified positions.        |
|      |`swap_line`        | Alias for `'swap_row'`.                                                |
|      |`swap_column`      | Convenience function to swap two columns at a specified positions.     |


### Reshaping:
|      |Function  |Description                                                                                                                                                    |
|------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      |`reshape` | Changes the matrix size to the one specified in argument. Throws an exception if the total number of elements in the new size does not match the current one. |

 
### Converting the matrix to STL vectors:
|          |Function        |Description                                                                                    |
|----------|----------------|-----------------------------------------------------------------------------------------------|
|***[v]*** | `to_vector_2D` | Converts the matrix to `std::vector<std::vector<T>>`.                                         |
|          | `to_vector_1D` | Converts the matrix to `std::vector<T>&` or `const std::vector<T>&` depending on the context. |

 
### Printing the matrix:
|          |Function     |Description                                                                                                    |
|----------|-------------|---------------------------------------------------------------------------------------------------------------|
|***[v]*** |`print`      | Prints the contents of the matrix in the specified std::ostream. If not specified, it prints it in std::cout. |
|***[v]*** |`print_size` | Prints the size of the matrix in the specified std::ostream. If not specified, it prints it in std::cout.     |

 
### Iterator-like functions:

|          |Function        |Description                                                                                                                               |
|----------|----------------|------------------------------------------------------------------------------------------------------------------------------------------|
|***[v]*** | `begin_row`    | Convenience function that returns `0`, to provide syntax as close as the one relative to `std::algorithm` as possible.                   |
|          | `begin_line`   | Alias for `'begin_row'`.                                                                                                                 |
|***[v]*** | `begin_column` | Convenience function that returns `0`, to provide syntax as close as the one relative to `std::algorithm` as possible.                   |
|***[v]*** | `end_row`      | Convenience function that returns the number of lines, to provide syntax as close as the one relative to `std::algorithm` as possible.   |
|          | `end_line`     | Alias for `'end_row'`.                                                                                                                   |
|***[v]*** | `end_column`   | Convenience function that returns the number of columns, to provide syntax as close as the one relative to `std::algorithm` as possible. |

 
### std::algorithm-like functions:
|      |Function           |Description                                                                                                                                  | Notes                                                            |
|------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
|      |`for_each`         | Analog to std::for_each. Applies the function in argument to every element in the matrix.                                                   | **WIP : Possibility to apply on a specific part of the matrix.** |
|      |`transform`        | Analog to std::transform. Applies the function in argument to every element in the matrix and modifies them according to its return value.  | **WIP : Possibility to apply on a specific part of the matrix.** |


### Sum, maximum, minimum, average:

|          |Function  |Description                                                                                                                                                              |
|----------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|***[v]*** |`sum`     | Returns the sum of all elements of the matrix, as `T` (meaning that overflow may occur).                                                                                |
|***[v]*** |`mean`    | Returns the mean value of all elements of the matrix, as `T` (meaning that rounding error and overflow may occur). It is computed as `sum() * (1./(rows()*columns()))`. |
|          |`average` | Alias for `mean()`.                                                                                                                                                     |
|***[v]*** | `max`    | Returns the maximum element (according to `std::max_element`) of the matrix.                                                                                            |
|***[v]*** | `min`    | Returns the minimum element (according to `std::min_element`) of the matrix.                                                                                            |

### Matrix-algebra related functions:
|          |Function         |Description                                                                                                                                                                      |
|----------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          |`trace`          | Returns the trace of the matrix, computed as `T` (meaning that rounding error and overflow may occur). Throws an exception (std::invalid_argument) if the matrix is not square. |
|***[v]*** |`det`            | Returns the determinant of the matrix. Throws an exception (`std::invalid_argument`) is the matrix is not square.                                                               |
|          |`cofactor`       | Returns the cofactor of the specified line and column or linear index. Throws an exception if one of them is outside the matrix.                                                |
|          |`comatrix`       | Returns the cofactor matrix. Convenience function that returns `cofactormatrix()`.                                                                                              |
|          |`cofactormatrix` | Returns the cofactor matrix.                                                                                                                                                    |
|          |`transpose`      | Returns the transpose of the matrix.                                                                                                                                            |
|          |`self_transpose` | Transposes current the matrix and returns a reference to it.                                                                                                                    |
|          |`inv`            | Returns the inverse of the matrix as computed by `operator!`.                                                                                                                   |
|          |`inverse`        | Returns the inverse of the matrix as computed by `operator!`.                                                                                                                   |
|**[S]**   |`invert`         | Returns the inverse of the matrix as computed by `operator!`.                                                                                                                   |
|**[S]**   |`hadamard`       | Returns the Hadamard product of two matrices. Throws an exception if the sizes do not match.                                                                                    |
|          |`element_wise`   | Convenience function that returns the Hadamard product of two matrices. Calls `hadamard`.                                                                                       |

### Matrix creation:
|        |Function      |Description                                                                      |
|--------|--------------|---------------------------------------------------------------------------------|
|        |`fill`        | Resizes the matrix as specified in argument and fills it with the value chosen. |
|**[S]** |`zeroes`      | Returns a matrix of the specified dimensions and filled with `T(0)`.            |
|**[S]** |`ones`        | Returns a matrix of the specified dimensions and filled with `T(1)`.            |
|**[S]** |`eye`         | Returns the identity matrix of the size specified in argument.                  |
|**[S]** |`randn`       | Creates a matrix of normally distributed numbers.                               |
|**[S]** |`uniform`     | Creates a matrix of uniformally distributed real numbers.                       |
|**[S]** |`uniform_int` | Creates a matrix of uniformally distributed integers.                           |
|**[S]** |`rand`        | Alias for `'uniform'`. Creates a matrix of uniformally distributed numbers.     |

 

 
### Operators:
|          |Function      |Description                                                                                                                                                                                                 |
|----------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          | `operator=`  | Assignment operator. Supports assignments from `std::vector<std::vector<T>>` and from other `Matrix`.                                                                                                      |
|                                                                                                                                                                                                                                      |
|          | `operator+`  | Computes the addition of a `Matrix` with another (term by term). Also supports the addition of value in `T`. In that case, it is the same as adding a Matrix of the same size holding only the same value. |
|          | `operator+=` | Adds arg to all the elements of the `Matrix`, and returns a reference to it.                                                                                                                               |
|                                                                                                                                                                                                                                      |
|***[v]*** | `operator*`  | Computes the usual matrix product of two matrices or multiplies term by term by the `T` speficied.                                                                                                         |
|***[v]*** | `operator*=` | Multiplies the matrix by the argument using the usual matrux product definition (or term by term if the argument is a `T`), and returns a reference to it.                                                 |
|                                                                                                                                                                                                                                      |
|          | `operator-`  | Computes the substraction of two Matrix or substacts the `T` argument to all elements.                                                                                                                     |
|          | `operator-=` | Computes the term by term difference of the matrix and the argument, or substracts the `T` in argument to all elements. Returns a reference to the current object.                                         |
|                                                                                                                                                                                                                                      |
|***[v]*** | `operator!`  | Returns the inverse of the matrix.                                                                                                                                                                         |
|                                                                                                                                                                                                                                      |
|***[v]*** | `operator^`  | Returns the matrix two the power specifed after the ^ (ex: a^2 returns `a*a`).                                                                                                                             |
|***[v]*** | `operator^=` | Raises the matrix to the specified power, and returns a reference to it.                                                                                                                                   |
|                                                                                                                                                                                                                                      |
|***[v]*** | `operator==` | Equality operator. Returns true only if both matrixes are of the same size, if all elements are identical and the same position.                                                                           |
|***[v]*** | `operator!=` | Returns the opposite of the result given by operator==.                                                                                                                                                    |

### Other non-member functions:


|      |Function           |Description                                                                                                                                  | Notes                                                            |
|------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
|      |`operator<<`       | Prints the matrix to the specified std::ostream.                                                                                            |                                                                  |
|      |`for_each`         | Analog to std::for_each. Applies the function in argument to every element in the matrix.                                                   | **WIP : Possibility to apply on a specific part of the matrix.** |
|      |`transform`        | Analog to std::transform. Applies the function in argument to every element in the matrix and modifies them according to its return value.  | **WIP : Possibility to apply on a specific part of the matrix.** |



# Using the GPU

To use the GPU for the calculations, you must `#define STE_MATRIX_ALLOW_GPU` first.
Then, you only need to specify on which device (CPU or GPU) the calculations involving a matrix will be made.
Example:
```c++

ste::IMatrix mat(28 , 28 , 0 , ste::EXE::CPU); //Calculations involving 'mat' will be made using the CPU unless it involves a matrix that requires to use the GPU.
const ste::FMatrix mat2(300 , 300 , 128.1f , ste::EXE::GPU); //ALL calculations involving 'mat2' will be made using the GPU.


```

***It is possible to change the execution policy during runtime :***

```c++

ste::FMatrix mat2(300 , 300 , 128.1f , ste::EXE::GPU); //ALL calculations involving 'mat2' will be made using the GPU.

//Do stuff here

mat2.device() = ste::EXE::CPU; //Now computations involving this matrix will be made using the CPU, except the ones involving a matrix with GPU execution policy.

//Do stuff here

mat2.device() = ste::EXE::GPU; //Analog to above, but the other way.

//...


```



On Windows, compiling CUDA requires you to :
    > Have it installed on your computer.
    > Use the Microsoft compiler (MSCV).

See **/documentation** for more details on how to use a CUDA-compatible GPU.

**A sample of a Qt .pro file is available in the 'qmake' folder.** 


# Upcoming features:

- Determinant calculated on GPU.
- Faster cofactormatrix.
- Transpose determined on GPU.
- Invert determined on GPU.

# Authors

 Developer / Tester : DUHAMEL Erwan (erwanduhamel@outlook.com)
 
 Tester : SOUDIER Jean (jean.soudier@insa-strasbourg.fr)
