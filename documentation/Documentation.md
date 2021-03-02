
# ste::Matrix class

C++ template class that provides an interface for matrix-based computing.


_This documentation layout is strongly inspired from the [Qt documentation](https://doc.qt.io/)._
_HTML export by [StackEdit](https://stackedit.io/)._


## Properties / Attributes

| Type            | Name      | Access            |
|-----------------|-----------|-------------------|
| `std::vector<T>`| \_data    | ***`protected`*** |
| `size_t`        | \_rows    | ***`protected`*** |
| `size_t`        | \_columns | ***`protected`*** |
| `ste::EXE`      | \_device  | ***`protected`*** |


## Public types

|             |                                                                                      |
|-------------|--------------------------------------------------------------------------------------|
| `enum class`| **[Orientation](#orientation_doc)**`{ ROW , LINE , RW , R , COLUMN , COL , CL , C }` |


## Public functions


|                                       |                                                                                                                                                                    |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                       | **[Matrix](#matrix_constructor_0)**`(const  size_t  &rows  ,  const  size_t& columns  ,  const  T  &value  =  T(0)  ,  const  EXE  &device  =  EXE::CPU)`          |
|                                       | **[Matrix](#matrix_constructor_1)**`(const  size_t  &size  =  0  ,  const  T  &value  =  T(0)  ,  const  EXE  &device  =  EXE::CPU)`                               |
|                                       | **[Matrix](#matrix_constructor_2)**`(const std::vector<T> &data , const size_t &rows , const size_t&columns , const EXE &device = EXE::CPU)`                       |
|                                       | **[Matrix](#matrix_constructor_3)**`(const std::vector<std::vector<T>> &data , const EXE &device = EXE::CPU)`                                                      |
| `Matrix&`                             | **[add](#add)**`(const std::vector<T> &data , const Orientation &orientation)`                                                                                     |
| `Matrix&`                             | **[add_row](#add_row)**`(const std::vector<T> &data)`                                                                                                              |
| `Matrix&`                             | **[add_line](#add_line)**`(const std::vector<T> &data)`                                                                                                            |
| `Matrix&`                             | **[add_column](#add_column)**`(const std::vector<T> &data)`                                                                                                        |
| `T`                                   | **[average](#average)**`() const`                                                                                                                                  |
| `T&`                                  | **[at](#at_0)**`(const size_t &row , const size_t &column)`                                                                                                        |
| `T&`                                  | **[at](#at_1)**`(const size_t &linear_index)`                                                                                                                      |
| `const T&`                            | **[at](#at_2)**`(const size_t &row , const size_t &column) const`                                                                                                  |
| `const T&`                            | **[at](#at_3)**`(const size_t &linear_index) const`                                                                                                                |
| `auto`                                | **[begin](#begin_0)**`() const`                                                                                                                                    |
| `auto`                                | **[begin](#begin_1)**`()`                                                                                                                                          |
| `virtual size_t`                      | **[begin_row](#begin_row)**`() const`                                                                                                                              |
| `size_t`                              | **[begin_line](#begin_line)**`() const`                                                                                                                            |
| `virtual size_t`                      | **[begin_column](#begin_column)**`() const`                                                                                                                        |
| `Matrix&`                             | **[clear](#clear)**`()`                                                                                                                                            |
| `T`                                   | **[cofactor](#cofactor_0)**`(const size_t &row , const size_t &column) const`                                                                                      |
| `T`                                   | **[cofactor](#cofactor_1)**`(const size_t &index) const`                                                                                                           |
| `Matrix`                              | **[cofactormatrix](#cofactormatrix)**`() const `                                                                                                                   |
| `Matrix`                              | **[comatrix](#comatrix)**`() const`                                                                                                                                |
| `std::vector<T>`                      | **[columnAt](#columnAt)**`(const size_t &index) const`                                                                                                             |
| `const size_t&`                       | **[columns](#columns)**`() const`                                                                                                                                  |
| `Matrix&`                             | **[cut](#cut_0)**`(std::vector<size_t> indexes , const Orientation &orientation)`                                                                                  |
| `Matrix&`                             | **[cut](#cut_1)**`(const size_t &begin , const size_t &end , const Orientation &orientation)`                                                                      |
| `Matrix&`                             | **[cut_columns](#cut_columns_0)**`(const std::vector<size_t> &indexes)`                                                                                            |
| `Matrix&`                             | **[cut_columns](#cut_columns_1)**`(const size_t &begin , const size_t &end)`                                                                                       |
| `Matrix&`                             | **[cut_lines](#cut_lines_0)**`(const std::vector<size_t> &indexes)`                                                                                                |
| `Matrix&`                             | **[cut_lines](#cut_lines_1)**`(const size_t &begin , const size_t &end)`                                                                                           |
| `Matrix&`                             | **[cut_rows](#cut_rows_0)**`(const std::vector<size_t> &indexes)`                                                                                                  |
| `Matrix&`                             | **[cut_rows](#cut_rows_1)**`(const size_t &begin , const size_t &end)`                                                                                             |
| `Matrix&`                             | **[deleteAll](#deleteAll)**`()`                                                                                                                                    |
| `virtual T`                           | **[det](#det)**`() const`                                                                                                                                          |
| `const EXE&`                          | **[device](#device_0)**`() const`                                                                                                                                  |
| `EXE&`                                | **[device](#device_1)**`()`                                                                                                                                        |
| `size_t`                              | **[elements](#elements)**`() const`                                                                                                                                |
| `Matrix`                              | **[element_wise](#element_wise)**`(const Matrix &arg) const`                                                                                                       |
| `bool`                                | **[empty](#empty)**`() const`                                                                                                                                      |
| `auto`                                | **[end](#end_0)**`()`                                                                                                                                              |
| `auto`                                | **[end](#end_1)**`() const`                                                                                                                                        |
| `virtual size_t`                      | **[end_column](#end_column)**`() const`                                                                                                                            |
| `size_t`                              | **[end_line](#end_line)**`() const`                                                                                                                                |
| `virtual size_t`                      | **[end_row](#end_row)**`() const`                                                                                                                                  |
| `Matrix&`                             | **[fill](#fill_0)**`(const size_t &size , const T &value)`                                                                                                         |
| `Matrix&`                             | **[fill](#fill_1)**`(const size_t &rows , const size_t &columns , const T &value)`                                                                                 |
| `Matrix&`                             | `template<class Function>`**[for_each](#for_each_0)**`(Function function)`                                                                                         |
| `const Matrix&`                       | `template<class Function>`**[for_each](#for_each_1)**`(Function function) const`                                                                                   |
| `Matrix`                              | **[hadamard](#hadamard)**`(const Matrix &arg) const`                                                                                                               |
| `Matrix&`                             | **[insert](#insert)**`(const size_t &element_index , const Orientation &orientation , const std::vector<T> &data)`                                                 |
| `Matrix&`                             | **[insert_column](#insert_column)**`(const size_t &index , const std::vector<T> &data)`                                                                            |
| `Matrix&`                             | **[insert_line](#insert_line)**`(const size_t &index , const std::vector<T> &data)`                                                                                |
| `Matrix&`                             | **[insert_row](#insert_row)**`(const size_t &index , const std::vector<T> &data)`                                                                                  |
| `Matrix`                              | **[inv](#inv)**`() const`                                                                                                                                          |
| `Matrix&`                             | **[invert](#invert)**`()`                                                                                                                                          |
| `bool`                                | **[isColumn](#isColumn)**`() const`                                                                                                                                |
| `bool`                                | **[isInvertible](#isInvertible)**`() const`                                                                                                                        |
| `bool`                                | **[isLine](#isLine)**`() const`                                                                                                                                    |
| `bool`                                | **[isRow](#isRow)**`() const`                                                                                                                                      |
| `bool`                                | **[isSquare](#isSquare)**`() const`                                                                                                                                |
| `std::vector<T>`                      | **[lineAt](#lineAt)**`(const size_t &index) const`                                                                                                                 |
| `const size_t&`                       | **[lines](#lines)**`() const`                                                                                                                                      |
| `T`                                   | **[max](#max)**`(std::function<T (const std::vector<T>&)> criterium = [](const std::vector<T> &data){return *std::max_element(data.begin() , data.end());}) const` |
| `virtual T`                           | **[mean](#mean)**`() const`                                                                                                                                        |
| `T`                                   | **[min](#min)**`(std::function<T (const std::vector<T>&)> criterium = [](const std::vector<T> &data){return *std::min_element(data.begin() , data.end());}) const` |
| `Matrix&`                             | **[operator=](#operator_equal_0)**`(const Matrix &arg)`                                                                                                            |
| `Matrix&`                             | **[operator=](#operator_equal_1)**`(const std::vector<std::vector<T>> &arg)`                                                                                       |
| `Matrix&`                             | **[operator=](#operator_equal_2)**`(const std::vector<T> &arg)`                                                                                                    |
| `virtual Matrix`                      | **[operator\*](#operator_mult_0)**`(const Matrix &arg) const`                                                                                                      |
| `Matrix`                              | **[operator\*](#operator_mult_1)**` (const T &arg) const`                                                                                                          |
| `virtual Matrix&`                     | **[operator\*=](#operator_mult_equal_0)**`(const Matrix &arg)`                                                                                                     |
| `Matrix&`                             | **[operator\*=](#operator_mult_equal_1)**`(const T &arg)`                                                                                                          |
| `Matrix`                              | **[operator+](#operator_plus_0)**`(const Matrix &arg) const`                                                                                                       |
| `Matrix`                              | **[operator+](#operator_plus_1)**`(const T &arg) const`                                                                                                            |
| `Matrix&`                             | **[operator+=](#operator_plus_equal_0)**`(const Matrix &arg)`                                                                                                      |
| `Matrix&`                             | **[operator+=](#operator_plus_equal_1)**`(const T &arg)`                                                                                                           |
| `Matrix`                              | **[operator-](#operator_minus_0)**`(const Matrix &arg) const`                                                                                                      |
| `Matrix`                              | **[operator-](#operator_minus_1)**`(const T &arg) const`                                                                                                           |
| `Matrix`                              | **[operator-](#operator_minus_2)**`() const`                                                                                                                       |
| `Matrix&`                             | **[operator-=](#operator_minus_equal_0)**`(const Matrix &arg)`                                                                                                     |
| `Matrix&`                             | **[operator-=](#operator_minus_equal_1)**`(const T &arg)`                                                                                                          |
| `virtual Matrix`                      | **[operator!](#operator_not)**`()const`                                                                                                                            |
| `virtual Matrix`                      | **[operator^](#operator_power)**`(const long long int &arg) const`                                                                                                 |
| `virtual Matrix&`                     | **[operator^=](#operator_power_equal)**`(const long long int &arg)`                                                                                                |
| `virtual bool`                        | **[operator==](#operator_equality)**`(const Matrix &arg) const`                                                                                                    |
| `virtual bool`                        | **[operator!=](#operator_difference)**`(const Matrix &arg) const`                                                                                                  |
| `virtual std::ostream&`               | **[print](#print)**`(std::ostream &outstream = std::cout) const`                                                                                                   |
| `virtual std::ostream&`               | **[print_size](#print_size)**`(std::ostream &outstream = std::cout) const`                                                                                         |
| `Matrix&`                             | **[push_back](#push_back)**`(const std::vector<T> &data , const Orientation &orientation)`                                                                         |
| `Matrix&`                             | **[push_back_column](#push_back_column)**`(const std::vector<T> &data)`                                                                                            |
| `Matrix&`                             | **[push_back_line](#push_back_line)**`(const std::vector<T> &data)`                                                                                                |
| `Matrix&`                             | **[push_back_row](#push_back_row)**`(const std::vector<T> &data)`                                                                                                  |
| `Matrix&`                             | **[remove](#remove)**`(const size_t &element_index , const Orientation &orientation)`                                                                              |
| `Matrix&`                             | **[remove_column](#remove_column)**`(const size_t &index)`                                                                                                         |
| `Matrix&`                             | **[remove_line](#remove_line)**`(const size_t &index)`                                                                                                             |
| `Matrix&`                             | **[remove_row](#remove_row)**`(const size_t &index)`                                                                                                               |
| `Matrix&`                             | **[replace](#replace_0)**`(const size_t &row , const unsigned &column , const T &value)`                                                                           |
| `Matrix&`                             | **[replace](#replace_1)**`(const size_t &index, const T &value)`                                                                                                   |
| `Matrix&`                             | **[replace](#replace_2)**`(const size_t &value_index ,const Orientation &orientation , const std::vector<T> &value)`                                               |
| `Matrix&`                             | **[replace](#replace_3)**`(const size_t &row_begin, const size_t &row_end ,const size_t &column_begin, const size_t &column_end,const T &value)`                 |
| `Matrix&`                             | **[replace_column](#replace_column)**`(const size_t &value_index , const std::vector<T> &value)`                                                                   |
| `Matrix&`                             | **[replace_line](#replace_line)**`(const size_t &value_index , const std::vector<T> &value)`                                                                       |
| `Matrix&`                             | **[replace_row](#replace_row)**`(const size_t &value_index , const std::vector<T> &value)`                                                                         |
| `Matrix&`                             | **[reshape](#reshape)**`(const size_t &rows , const size_t &columns)`                                                                                              |
| `std::vector<T>`                      | **[rowAt](#rowAt)**`(const size_t &index) const`                                                                                                                   |
| `const size_t&`                       | **[rows](#rows)**`() const`                                                                                                                                        |
| `Matrix&`                             | **[self_transpose](#self_transpose)**`()`                                                                                                                          |
| `Matrix&`                             | **[setDevice](#setDevice)**`(const EXE &device)`                                                                                                                   |
| `const std::vector<size_t>`           | **[size](#size)**`() const`                                                                                                                                        |
| `virtual T`                           | **[sum](#sum)**`() const`                                                                                                                                          |
| `Matrix&`                             | **[swap](#swap)**`(const size_t &element_1 , const size_t &element_2 ,const Orientation &orientation)`                                                             |
| `Matrix&`                             | **[swap_columns](#swap_columns)**`(const size_t &element_1 , const size_t &element_2)`                                                                             |
| `Matrix&`                             | **[swap_lines](#swap_lines)**`(const size_t &element_1 , const size_t &element_2)`                                                                                 |
| `Matrix&`                             | **[swap_rows](#swap_rows)**`(const size_t &element_1 , const size_t &element_2)`                                                                                   |
|`const std::vector<T>&`                | **[toVector1D](#toVector1D_0)**`() const`                                                                                                                          |
|`std::vector<T>&`                      | **[toVector1D](#toVector1D_1)**`()`                                                                                                                                |
| `virtual std::vector<std::vector<T>>` | **[toVector2D](#toVector2D)**`() const`                                                                                                                            |
| `T`                                   | **[trace](#trace)**`() const`                                                                                                                                      |
| `Matrix&`                             | `template<class Function>`**[transform](#transform)**`(Function function)`                                                                                         |
| `Matrix`                              | **[transpose](#transpose)**`() const`                                                                                                                              |
| `Matrix&`                             | **[transpose_in_place](#transpose_in_place)**`()`                                                                                                                  |


## Static public members

|          |                                                                                                                                                                |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Matrix` | **[eye](#eye)** `(const size_t &size , const EXE &device = EXE::CPU)`                                                                                          |
| `Matrix` | **[ones](#ones_0)**`(const size_t &size , const EXE &device = EXE::CPU)`                                                                                       |
| `Matrix` | **[ones](#ones_1)**`(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`                                                               |
| `Matrix` | **[uniform](#uniform_0)**`(const size_t &size , const EXE &device = EXE::CPU)`                                                                                 |
| `Matrix` | **[uniform](#uniform_1)**`(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`                                                         |
| `Matrix` | **[uniform_int](#uniform_int_0)**`(const size_t &size , const EXE &device = EXE::CPU)`                                                                         |
| `Matrix` | **[uniform_int](#uniform_int_1)**`(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`                                                 |
| `Matrix` | **[rand](#rand_0)**`(const size_t &size , const EXE &device = EXE::CPU)`                                                                                       |
| `Matrix` | **[rand](#rand_1)**`(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`                                                               |
| `Matrix` | **[randn](#randn_0)**`(const size_t &size , const T &mean = T(0) , const T &standard_deviation = T(1.) const EXE &device = EXE::CPU)`                          |
| `Matrix` | **[randn](#randn_1)**`(const size_t &rows , const size_t &columns , const T &mean = T(0) , const T &standard_deviation = T(1) , const EXE &device = EXE::CPU)` |
| `Matrix` | **[zeroes](#zeroes_0)**`(const size_t &size , const EXE &device = EXE::CPU)`                                                                                   |
| `Matrix` | **[zeroes](#zeroes_1)**`(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`                                                           |

## Non-member functions

|                       |                                                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| `Matrix`              | `template<class T>`**[element_wise](#element_wise_nm)**`(const Matrix<T> &arg1 , const Matrix<T> &arg2)`          |
| `Matrix&`             | `template<class T , class Function>`**[for_each](#for_each_nm_0)**`(Matrix<T> &matrix , Function function)`       |
| `const Matrix&`       | `template<class T , class Function>`**[for_each](#for_each_nm_1)**`(const Matrix<T> &matrix , Function function)` |
| `Matrix`              | `template<class T>`**[hadamard](#hadamard_nm)**`(const Matrix<T> &arg1 , const Matrix<T> &arg2)`                  |
| `Matrix&`             | `template<class T>`**[invert](#invert_nm)**`(Matrix<T> &arg)`                                                     |
| `std::ostream&`       | `template<class T>`**[operator\<\<](#operator_print_mat_nm)**`(std::ostream &outstream , const Matrix<T> &arg)`   |
| `std::ostream&`       | **[operator\<\<](#operator_print_EXE_nm)**`(std::ostream &outstream , const EXE &a)`                              |
| `EXE`                 | **[operator\|](#operator_binary_or_EXE_nm)**`(const EXE &a , const EXE &b)`                                       |
| `bool`                | **[operator\|\|](#operator_logical_or_EXE_nm)**`(const EXE &a , const EXE &b)`                                    |
| `EXE`                 | **[operator&](#operator_binary_and_EXE_nm)**`(const EXE &a , const EXE &b)`                                       |
| `bool`                | **[operator&&](#operator_logical_and_EXE_nm)**`(const EXE &a , const EXE &b)`                                     |
| `Matrix&`             | `template<class T , class Function>`**[transform](#transform_nm)**`(Matrix<T> &matrix , Function function)`       |

## Non-member types

|                                      |                                                                                 |
|--------------------------------------|---------------------------------------------------------------------------------|
| `enum class`                         | **[EXE](#EXE)**`{CPU = 0, C = CPU, HOST = CPU, GPU = 1, G = GPU, DEVICE = GPU}` |
| `typedef Matrix<float>`              | **[FMatrix](#FMatrix)**                                                         |
| `typedef Matrix<double>`             | **[DMatrix](#DMatrix)**                                                         |
| `typedef Matrix<long double>`        | **[LDMatrix](#LDMatrix)**                                                       |
| `typedef Matrix<int>`                | **[IMatrix](#IMatrix)**                                                         |
| `typedef Matrix<long int>`           | **[LIMatrix](#LIMatrix)**                                                       |
| `typedef Matrix<long long int>`      | **[LLIMatrix](#LLIMatrix)**                                                     |
| `typedef Matrix<unsigned int>`       | **[UIMatrix](#UIMatrix)**                                                       |
| `typedef Matrix<unsigned long>`      | **[ULMatrix](#ULMatrix)**                                                       |
| `typedef Matrix<unsigned long long>` | **[ULLMatrix](#ULLMatrix)**                                                     |
| `typedef Matrix<char>`               | **[CMatrix](#CMatrix)**                                                         |
| `typedef Matrix<unsigned char>`      | **[UCMatrix](#UCMatrix)**                                                       |

## Macros

|          |                                                   |
|----------|---------------------------------------------------|
| No value | **[STE_MATRIX_ALLOW_GPU](#ste_matrix_allow_gpu)** |


## Detailed Description

The `ste::Matrix` template class offers an interface for any matrix computation-related applications.
 
Simple to use and highly customizable through inheritance, this class provides high level functions to fasten the developpment.
Among its main features:
 >• CUDA-compatible. Refer to the paragraph **[Using a GPU](#using-a-gpu)** for more information.
 >• Dynamic reshaping and resizing.
 >• Determinant, trace, sum, average, min, max...
 >• Cofactors, cofactor matrix, inverse, transpose, Hadamard (element-wise) product.
 >• Multiplication, sum, difference operators.

<br>

### Initializing a ste::Matrix

This class offers four different constructors for convenience purposes.

The following code snippets creates a `2*4` float matrix filled with 0, with CPU as default execution policy.
```c++
    ste::Matrix<float> mat(2 , 4 , 0 , ste::EXE::CPU); //2*4 float matrix filled with 0.
   /* 
    mat is :
        0 0 0 0
        0 0 0 0
    */
    
```

To create a square matrix, proceed this way :
```c++
    ste::Matrix<int> mat_square(2 , 99 , ste::EXE::CPU); //2*2 int square matrix filled with 99.
    
    /* 
    mat_square is :
        99 99
        99 99
    */
```

It is possible to speficy the data when creating the matrix, as long as
```c++
    ste::Matrix<unsigned> mat_from_vector({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); //3*3 unsigned matrix with CPU as default execution policy.
    
    /* 
    mat_from_vector is :
        1 2 3
        4 5 6
        7 8 9
    */
```

Finally, if your data is stored in a `std::vector<std::vector<T>>`, simply proceed this way ;
**To use this constructor, all `std::vector<T>` must be of the same size.**

```c++
    ste::Matrix<double> mat_from_vector2D({{1 , 2 , 3} , {4 , 5 , 6}} , ste::EXE::GPU); //2*3 double matrix that will use the GPU (refere to the dedicated paragraph for more details).

    /* 
    mat_from_vector2D is :
        1 2 3
        4 5 6
    */
```


<br>

### Manipulating a ste::Matrix data

#### Accessing an element

Element access is made possible using `at`. It **always** returns a reference to the element, meaning that you can modify it this way.
 
Both linear indexes and `(row, column)` coordinates are acceptable arguments.
 
**The two methods performances are identical.**

```c++
    ste::Matrix<unsigned> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3);

    std::cout << mat.at(3) << std::endl;     //Prints 4
    std::cout << mat.at(2 , 2) << std::endl; //Prints 5

```


To access a row or a column, simply use `rowAt` , `lineAt` or `columnAt`.

```c++
    ste::Matrix<unsigned> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); 

    mat.rowAt(2);    //Returns std::vector<unsigned>({6 , 7 , 8})
    mat.columnAt(1): //Returns std::vector<unsigned>({2,5,8})

```


#### Modifying the matrix shape

##### Basic reshaping

`ste::Matrix` provides `O(1)` reshaping **as long as data is not appended or removed**.
 
`reshape` changes the size of the matrix, and throws an exception if the total number of element do not match.

```c++
    ste::Matrix<unsigned> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); 

    mat.reshape(1,9); //mat is now 1*9 matrix.
    mat.reshape(9,1); //mat is now 9*1 matrix.

```

##### Appending data to the matrix

To add column or row, using `add` or `push_back` as follow :

```c++
    ste::Matrix<float> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); 

    mat.add({10 , 11 , 12} , ste::FMatrix::Orientation::ROW); //Adds a row at the end of the matrix.
    ///This syntaxe being heavy, it is recommended to use either 'push_back_row' or 'add_row'.

    /*
        mat is now :

        1   2   3
        4   5   6
        7   8   9
        10  11  12
    */
    
    mat.push_back_column({100 , 101 , 102 , 103});

    /*
        mat is now :

        1   2   3   100
        4   5   6   101
        7   8   9   102
        10  11  12  103
    */

```

##### Inserting data

You can insert a row or a column using the `insert` functions.
 
**The sizes must match when appending to a matrix**.
 
```c++
    ste::Matrix<double> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); 

    mat.insert_row(0 , {10 , 11 , 12}); //Inserts a row at position '0'

    /*
        mat is now :

        10  11  12
        1   2   3
        4   5   6
        7   8   9

    */

    mat.insert_column(1 , {100 , 101 , 102 , 103});  //Inserts a column at position '1'

    /*
        mat is now :

        10  100  11  12
        1   101  2   3
        4   102  5   6
        7   103  8   9

    */

```

##### Removing data

You can remove a single row or a single column using the `remove` functions.


```c++

    ste::Matrix<double> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); 

    mat.remove_row(0); //Removes the first row

    /*
        mat is now :

        4   5   6
        7   8   9
    */


    mat.remove_column(0); //Removes the first column

    /*
        mat is now :

         5   6
         8   9
    */
```

To remove several rows or columns at one, use `cut` as follows :

```c++

    ste::Matrix<double> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); 
    
    mat.cut_rows({2 , 0}); //Removes rows 0 and 2.
    
    /*
       mat is now :
        
        4   5   6
    */
    
    mat.cut_columns(0 , 1); //Removes columns 0 to 1 (included).
   /*
        mat is now :
          6
    */

```

##### Swapping elements


You can swap two rows or columns using the `swap` functions.


```c++

    ste::Matrix<double> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); 
    
    mat.swap_rows(2 , 0); //Swaps rows 0 and 2.
    
    /*
       mat is now :
        7   8   9
        4   5   6
        1   2   3
    */
    
    mat.swap_columns(1 , 2); //Swaps columns 1 and 2.
   /*
        mat is now :
        7   9   8
        4   6   5
        1   3   2
    */

```


##### Replacing data

Should you need to replace an entire column or entire row, use the `replace` functions.


```c++

    ste::Matrix<float> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); 

    mat.replace_row(0 , {99 , 98 , 97});
    /*
        mat is now :
        99  98  97
        4   5   6
        7   8   9
    */

    mat.replace_column(2 , {100 , 101 , 102});
    /*
        mat is now :
        99  98  100
        4   5   101
        7   8   102
    */

```

##### Filling a matrix

You can completely override a matrix content and change its size using `fill`.

**WARNING!** When the contents are dynamically allocated (pointers), memory **IS NOT FREED**.

```c++

    ste::Matrix<float> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3);

    mat.fill(8 , 22);   //mat is now a 8*8 float matrix holding 22 in each element.

    mat.fill(3,2, 1.6); //mat is now a 3*2 float matrix holding 1.6 in each element.

```

#### Applying a function to all elements

The `ste` namespace provides the analogs functions to `std::for_each` and `std::transform` for any instantiation of a `ste::Matrix`.
 
For convenience purposes, `ste::Matrix` also has member functions named `for_each` and `transform`, that behave the same.

```c++

    ste::Matrix<float> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3);

    ste::for_each(mat , [](const float &value){std::cout << value << " ";});
    /*
    Identical to :
        mat.for_each([](const float &value){std::cout << value << " ";});
        
        Prints '1 2 3 4 5 6 7 8 9'.
    */


    ste::transform(mat , [](const float &value){return 2*value;});
    /*
    Identical to :
        mat.transform([](const float &value){return 2*value;});
        
        mat is now :
        2  4  6
        8  10 12
        14 16 18

    */
```

<br>

### Printing a ste::Matrix


You can print a `ste::Matrix` to any `std::ostream` by using either `print() const` or `operator<<`.
 
Note that `print() const` is `virtual`.


```c++

    ste::Matrix<float> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3);

    mat.print() << std::endl; //Prints "[ [ 1 2 3 ] [ 4 5 6 ] [ 7 8 9 ] ]" to std::cout.
    std::cout << mat << std::endl; //Same as above.

    mat.print(std::clog); //Prints the matrix contents to std::clog.
    std::clog << mat; //Same as above.

```

To print a matrix size without having to write anything yourself, use `print_size() const`.

```c++

    ste::Matrix<float> mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3);

    mat.print_size() << std::endl;           //Prints "[3 ; 3]" to std::cout.

    mat.print_size(std::cerr) << std::endl;  //Prints "[3 ; 3]" to std::cerr.

```

<br>

### Maths-related functions


`ste::Matrix` provides several convenience maths functions.



You can for example compute the numerical value of the determinant of any square matrix by calling `det` :

```c++

    const ste::FMatrix square_mat({1 , 2 , 3 , 4 , 52 , 6 , 7 , 141 , 9} , 3 , 3); //3*3 float matrix

    std::cout << square_mat.det() << std::endl;     //Prints the determinant (here 234) to std::cout

```



Should you need the explicit expression of the inverse of a matrix, you can use `operator!`, `inv` or `invert`.

**Explicitely computing a matrix inverse is HEAVY. You should not try to compute inverses of matrices bigger than 10 x 10.**

```c++

    ste::FMatrix square_mat({1 , 2 , 3 , 4 , 52 , 6 , 7 , 141 , 9} , 3 , 3); //3*3 float matrix

    /*
    This matrix inverse is: 
        -1.61538   1.73077     -0.615385 
        0.025641   -0.0512821   0.025641
        0.854701   -0.542735    0.188034 
    
    */


    std::cout << !square_mat << std::endl;              //Computes and prints the inverse of square_mat to std::cout
    std::cout << square_mat.inv() << std::endl;         //Equivalent to above
    std::cout << square_mat.invert() << std::endl;      //square_mat is now !square_mat.

```

You can also compute any cofactor and the cofactor matrix explicitely :

**Computing the cofactor matrix is the same complexity as computing the inverse.**

```c++

    const ste::FMatrix square_mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); //3*3 float matrix

    std::cout << square_mat.cofactor(0) << std::endl;         //Cofactor using linear indexes
    std::cout << square_mat.cofactor(0,1) << std::endl;       //Cofactor using (x,y) indexes

    std::cout << square_mat.cofactormatrix() << std::endl;    //Prints the cofactor matrix

```

Among the other features : `trace` , `max` / `min` , `sum` and `average` :


```c++

    const ste::FMatrix square_mat({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3); //3*3 float matrix

    std::cout << square_mat.trace() << std::endl;   //Prints the trace of a matrix (the sum of the diagonal terms)

    std::cout << square_mat.max() << std::endl;      //Prints the maximum of the matrix. Default uses 'std::max_element'
    std::cout << square_mat.min() << std::endl;      //Prints the minimum of the matrix. Default uses 'std::min_element'
    std::cout << square_mat.sum() << std::endl;      //Prints the sum of all the elements of the matrix.
    std::cout << square_mat.average() << std::endl;  //Prints the average of the matrix.

```

**If your matrix does not contain a POD type, you can add a criterium to `min` and `max`. See their documentation for more details.**

<br>

### Converting to `std::vector` or `std::vector<std::vector>>`


Especially using CUDA, it is often necessary to be able to obtain a `std::vector` holding the data.

`ste::Matrix` provides such a function :

```c++

    const ste::FMatrix mat_1({1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9} , 3 , 3);   //3*3 float matrix
    ste::ULLMatrix mat_2({10 , 11 , 12 ,13 ,14 , 15 , 16 , 17 , 18} , 3 , 3); //3*3 uint64_t matrix

    mat_1.toVector1D(); //Returns a const std::vector<float>& holding {1 , 2 , 3 ,4 ,5 , 6 , 7 , 8 , 9}
    mat_2.toVector1D(); //Returns a std::vector<uint64_t>& holding {10 , 11 , 12 ,13 ,14 , 15 , 16 , 17 , 18} 

```

**As the data is passed by reference, it is possible to modify it using `toVector1D()` when the matrix is not `const`.**


```c++

    ste::FMatrix a({1 , 2 , 3 ,
                    4 , 52 , 6 ,
                    7 , 141 , 9} ,
                                  3 , 3); //3*3 float matrix


    std::vector<float> &vect = a.toVector1D();
    vect.at(0) = 999;

    std::cout << a << std::endl; //Prints [ [ 999 2 3 ] [ 4 52 6 ] [ 7 141 9 ] ]

```

Should you need a `std::vector<std::vector<T>>`, simply use `toVector2D` :

```c++

    ste::FMatrix a({1 , 2 , 3 ,
                    4 , 52 , 6 ,
                    7 , 141 , 9} ,
                                  3 , 3); //3*3 float matrix

    std::vector<std::vector<float>> two_dimensional_vector = a.toVector2D();

    /*
        two_dimensional_vector is :

         { {1 ,  2 ,  3} ,  //First vector
           {4 ,  52 , 6} ,  //Second vector
           {7 , 141 , 9}    //Third vector
           }
    */

```

`toVector2D` orders the data **by row** . This method is `virtual`, allowing you to change the format should you need it.

<br>

### Using a GPU


#### Windows - ***Qt***

Configuring Qt on Windows for using CUDA librairies requires you to use **MSVC**.

**Requirements:**
*1.* Make sure your computer is equipped with a CUDA-compatible GPU.
*2.* Download the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
*3.* Download Visual Studio and a version of **MSVC** compatible with the CUDA Toolkit version.


**ste::Matrix specific requirements:**

Add `STE_MATRIX_ALLOW_GPU` to the `DEFINES` of your `.pro file`:

```makefile
DEFINES += STE_MATRIX_ALLOW_GPU
```

<br>

**Writing the `.pro` file:**

This tutorial is based on [this GitHub repository](https://github.com/mihaits/Qt-CUDA-example) by [mihaits](https://github.com/mihaits).


1. CUDA and MSVC have conflicting names in some of their functions. Adding these lines to the `.pro` file solves the problem:


```makefile
# Avoid conflicts between CUDA and MSVC
QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:msvcrtd.lib
QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:libcmt.lib



# Avoid conflicts between CUDA and MSVC
QMAKE_CFLAGS_DEBUG      += /MTd
QMAKE_CFLAGS_RELEASE    += /MT
QMAKE_CXXFLAGS_DEBUG    += /MTd
QMAKE_CXXFLAGS_RELEASE  += /MT
```

2. Create a list of the CUDA source files for your project:

```makefile
CUDA_SOURCES +=  ../someDistantFolder/some_CUDA_code.cu \
                 someFolder/other_CUDA_code.cu
```

3. Specify the CUDA installation path, the system architecture, the system type (32 bits or 64 bits), the architecture of your GPU, and the CUDA compiler options:

```makefile
CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1" #Path to your CUDA installation
SYSTEM_NAME = x64                                                     #NB: SYSTEM DEPENDENT
SYSTEM_TYPE = 64                                                      #SAME HERE
CUDA_ARCH = sm_61                                                     #Compute capability of the GPU (here GTX Geforce 1050 - Compute capability 6.1 so sm_61)
NVCC_OPTIONS = --use_fast_math                                        #CUDA compiler options
```

4. Include CUDA headers and CUDA libraires (such as CUBLAs) headers:

```makefile
#CUDA Headers
INCLUDEPATH += $$CUDA_DIR/include

#CUDA librairies headers
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME

#Required libraires added here
LIBS += -lcuda -lcudart -lcublas

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
```

5. Configure the CUDA compiler:

```makefile
#Configuration of the CUDA compiler
CONFIG(debug, debug|release) {
       # Debug mode
       cuda_d.input = CUDA_SOURCES
       cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
       cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
       cuda_d.dependency_type = TYPE_C
       QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
       # Release mode
       cuda.input = CUDA_SOURCES
       cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
       cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
       cuda.dependency_type = TYPE_C
       QMAKE_EXTRA_COMPILERS += cuda
}
```

6. Known troubles:

- Debug build sometimes fails with some CUDA librairies.


**ste::Matrix specific parts:**

1. Add the CUDA-related headers to the list of the headers:

```makefile
HEADERS += ../Matrix/CUDA_src/CUDA_global.h \
           ../Matrix/CUDA_src/CUDA_matrix_operators.h \
```

2. Add the CUDA-related source files to the list of the CUDA files:

```makefile
CUDA_SOURCES +=  ../Matrix/CUDA_src/CUDA_global.cu \
                 ../Matrix/CUDA_src/CUDA_matrix_operators.cu \
                 ../Matrix/CUDA_src/CUDA_setup.cu \
```


<br>

For a sample `.pro` file, see [/qmake](https://github.com/Erellu/ste-Matrix/tree/master/qmake).

<br>


#### Windows - ***Visual Studio***

TO BE ADDED

<br>

#### Ubuntu

TO BE ADDED

<br>

#### When to use a GPU ?


Using your GPU for all computations is **not** recommended.
As its RAM and the CPU one are not shared, it is required to copy data two times, meaning that bottleneck effects occur.
**Small matrix-related computations run faster on a CPU than on a GPU**.

For further information, you may look at the following sites:
- [When should I be offloading work to a GPU instead of the CPU?](https://softwareengineering.stackexchange.com/questions/107416/when-should-i-be-offloading-work-to-a-gpu-instead-of-the-cpu)
- [CPU vs. GPU: Making the Most of Both](https://www.intel.com/content/www/us/en/products/docs/processors/cpu-vs-gpu.html)
- [What’s the Difference Between a CPU and a GPU?](https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/)


<br>
<br>




## Public types documentation

<br>

<div id = "orientation_doc"></div>

### `enum class` Matrix::Orientation

| Constant            | Value                 | Description                                                                                            |
| --------------------|-----------------------|--------------------------------------------------------------------------------------------------------|
|`Orientation::ROW`   |   0                   | Specifier for functions that apply either on rows or on columns, such as `replace`, `swap` or `remove`.|
|`Orientation::LINE`  | `Orientation::ROW`    | Alias for `Orientation::ROW`                                                                           |
|`Orientation::RW`    | `Orientation::ROW`    | Alias for `Orientation::ROW`                                                                           |
|`Orientation::R`     | `Orientation::ROW`    | Alias for `Orientation::ROW`                                                                           |
|`Orientation::COLUMN`|   1                   | Specifier for functions that apply either on rows or on columns, such as `replace`, `swap` or `remove`.|
|`Orientation::COL`   | `Orientation::COLUMN` | Alias for `Orientation::COLUMN`                                                                        |
|`Orientation::CL`    | `Orientation::COLUMN` | Alias for `Orientation::COLUMN`                                                                        |
|`Orientation::C`     | `Orientation::COLUMN` | Alias for `Orientation::COLUMN`                                                                        |









## Member Functions Documentation

<br>
<div id="matrix_constructor_0"></div>

### **Matrix**`(const size_t &rows , const size_t&columns , const T &value = T(0) , const EXE &device = EXE::CPU)`

Constructs a matrix of size `rows * columns` filled with `value` and with execution policy `device`.

Default fill value is `T(0)`.
Default execution policy is `EXE::CPU`.



<br>
<div id="matrix_constructor_1"></div>

### **Matrix**`(const size_t &size = 0 , const T &value = T(0) , const EXE &device = EXE::CPU)`

Constructs a square matrix of size `size` filled with `value` and with execution policy `device`.

Default fill value is `T(0)`.
Default execution policy is `EXE::CPU`.


**This is the default constructor called by `Matrix()`, which constructs an empty matrix (0 by 0).**

<br>
<div id="matrix_constructor_2"></div>

### **Matrix**`(const std::vector<T> &data , const size_t &rows , const size_t&columns , const EXE &device = EXE::CPU)`

Constructs a matrix from the elements of `data`.

***Throws an exception*** if `rows * columns != data.size()`.



<br>
<div id="matrix_constructor_3"></div>

### **Matrix**`(const std::vector<std::vector<T>> &data , const EXE &device = EXE::CPU)`

Constructs a matrix from a two dimensional vector.

***Throws an exception*** if the size of all the vectors is not identical.



<br>
<div id="add"></div>

### `Matrix&` **add**`(const std::vector<T> &data , const Orientation &orientation)`

Adds `data` as a row or a column at the end of the matrix according to `orientation`.

***Throws an exception*** if the number of elements in `data` do not match the matrix dimensions.


<br>
<div id = "add_column"></div>

### `Matrix&` **add_column**`(const std::vector<T> &data)`

Adds `data` as a column at the end of the matrix.
 
***Throws an exception*** if the number of elements in `data` do not match the matrix dimensions.

```c++

    mat.add(data , ste::Matrix<T>::Orientation::COLUMN);

```

<br>
<div id = "add_line"></div>

### `Matrix&` **add_line**`(const std::vector<T> &data)`

Alias for **[add_row](#add_row)**.


<br>
<div id = "add_row"></div>

### `Matrix&` **add_row**`(const std::vector<T> &data)`

Adds `data` as a row at the end of the matrix.

***Throws an exception*** if the number of elements in `data` do not match the matrix dimensions.


This function is identical to:

```c++

    mat.add(data , ste::Matrix<T>::Orientation::ROW);

```

<br>
<div id = "average"></div>

### `T` **average**`() const`

Computes the average of the matrix.
 
Implementation gives :

```c++

    mat.average() == mat.sum() * (1./(mat.elements()));

```


<br>
<div id = "at_0"></div>

### `T&` **at**`(const size_t &row , const size_t &column)`

Returns a reference to the element at the `(x,y)` position specified in argument.
 
***Throws an exception*** if the position is outside the matrix.
 
This is identical to calling :

```c++

    mat.at(row *_columns + column).

```


<br>
<div id = "at_1"></div>

### `T&` **at**`(const size_t &linear_index)`

Returns a reference to the element at the linear index specified in argument.
 
***Throws an exception*** if the position is outside the matrix.


<br>

<div id = "at_2"></div>

### `const T&` **at**`(const size_t &row , const size_t &column)`

Returns a reference to the element at the `(x,y)` position specified in argument.
 
***Throws an exception*** if the position is outside the matrix.


<br>
<div id = "at_3"></div>

### `const T&` **at**`(const size_t &linear_index)`

Returns a reference to the element at the linear index specified in argument.
 
***Throws an exception*** if the position is outside the matrix.



<br>

<div id = "begin_0"></div>

### `auto` **begin**`() const`

Returns an iterator to the beginning of the data of the matrix.


<br>
<div id = "begin_1"></div>

### `auto` **begin**`()`

Returns an iterator to the beginning of the data of the matrix.
 
Non-const equivalent of **[begin](#begin_0)**.


<br>
<div id = "begin_row"></div>

### `virtual size_t` **begin_row**`() const`

Provided for specifying were the rows begin in the data.
 
Default implementation returns `0`.


<br>
<div id = "begin_line"></div>

### `size_t` **begin_line**`() const`

Alias for **[begin_row](#begin_row)**


<br>
<div id = "begin_column"></div>

### `virtual size_t` **begin_column**`() const`

Provided for specifying were the columns begin in the data.
 
Default implementation returns `0`.



<br>
<div id = "clear"></div>

### `Matrix&` **clear**`()`

Resizes the matrix to `(0 ; 0)` and removes all the elements from the data it holds.
 
**WARNING : MEMORY IS NOT FREED WHEN T IS DYNAMICALLY ALLOCATED.**


<br>
<div id = "cofactor_0"></div>

### `T` **cofactor**`(const size_t &row , const size_t &column) const`

Computes the cofactor of the element speficied in argument.
 
***Throws an exception*** if the position is outside the matrix.


<br>
<div id = "cofactor_1"></div>

### `T` **cofactor**`(const size_t &index) const`

Computes the cofactor of the element speficied in argument.
 
***Throws an exception*** if the position is outside the matrix.

This is equivalent to calling:

```c++

    mat.cofactor(index / _columns , index % _columns);

```





<br>
<div id = "cofactormatrix"></div>

### `Matrix` **cofactormatrix**`() const`

Computes the cofactor matrix.


<br>
<div id = "comatrix"></div>

### `Matrix` **comatrix**`() const`

Alias for **[cofactormatrix](#cofactormatrix)**.


<br>
<div id = "columnAt"></div>

### `std::vector<T>` **columnAt**`(const size_t &index) const`

Extracts a column of the matrix and returns it as a `std::vector<T>`.
 
***Throws an exception*** if the position is outside the matrix.

<br>
<div id = "columns"></div>

### `const size_t&` **columns**`() const`

Returns the number of columns of the matrix.


<br>
<div id = "cut_0"></div>

### `Matrix&` **cut**`(std::vector<size_t> indexes , const Orientation &orientation)`

Removes all the rows or columns according to the indexes speficied in `indexes`.
 
***Throws an exception*** if one the positions is outside the matrix.

<br>
<div id = "cut_1"></div>

### `Matrix&` **cut**`(const size_t &begin , const size_t &end , const Orientation &orientation)`

Removes all the rows or columns from `begin` to `end` (included).
 
***Throws an exception*** if one the positions is outside the matrix.


<br>
<div id = "cut_columns_0"></div>

### `Matrix&` **cut_columns**`(const std::vector<size_t> &indexes)`

Removes all the columns specified in `indexes`.
 
***Throws an exception*** if one the positions is outside the matrix.

This is identical to calling:

```c++

    mat.cut(indexes , ste::Matrix<T>::Orientation::COLUMN);

```

<br>

<div id = "cut_columns_1"></div>

### `Matrix&` **cut_columns**`(const size_t &begin , const size_t &end)`

Removes all the columns between `begin` and `end` (included).

***Throws an exception*** if one the positions is outside the matrix.

This is identical to calling:

```c++

    mat.cut(begin , end , ste::Matrix<T>::Orientation::COLUMN);

```

<br>
<div id = "cut_lines_0"></div>

### `Matrix&` **cut_lines**`(const std::vector<size_t> &indexes)`

Alias for **[cut_rows](#cut_rows_0)**.

<br>

<div id = "cut_lines_1"></div>

### `Matrix&` **cut_lines**`(const size_t &begin , const size_t &end)`

Alias for **[cut_rows](#cut_rows_1)**.

<br>

<div id = "cut_rows_0"></div>

### `Matrix&` **cut_rows**`(const std::vector<size_t> &indexes)`

Removes all the rows specified in `indexes`.
 
***Throws an exception*** if one the positions is outside the matrix.

This is identical to calling:

```c++

    mat.cut(indexes , ste::Matrix<T>::Orientation::ROW);

```



<br>
<div id = "cut_rows_1"></div>

### `Matrix&` **cut_rows**`(const size_t &begin , const size_t &end)`

Removes all the rows between `begin` and `end` (included).

***Throws an exception*** if one the positions is outside the matrix.

This is identical to calling:

```c++

    mat.cut(begin , end , ste::Matrix<T>::Orientation::ROW);

```

<br>
<div id = "deleteAll"></div>

### `Matrix&` **deleteAll**`()`

Frees the memory of all elements contained by the matrix, which is resized to `(0 ; 0)`.

***This member function is only available when T is dynamically allocated.***




<br>
<div id = "det"></div>

### `virtual T` **det**`() const`

Computes the determinant of the matrix.

***Throws an exception*** if the matrix is not squared.


*If your matrix has specific properties that simplify the computation of the determinant, it is recommended to override this function in a subclass.*


<br>
<div id = "device_0"></div>

### `const EXE&` **device**`() const`

Returns a reference to the execution policy of the matrix.

<br>

<div id = "device_1"></div>

### `EXE&` **device**`()`

Returns a reference to the execution policy of the matrix.

<br>

<div id = "elements"></div>

### `size_t` **elements**`() const`

Returns the total of elements stored in a matrix.

**As a `ste::Matrix` is always contiguous**, `elements` returns `_rows * columns`.


<br>
<div id = "element_wise"></div>

### `Matrix` **element_wise**`(const Matrix &arg) const`

Alias for **[hadamard](#hadamard)**.

Computes the Hadamard product (element-wise product) of the matrix and the argument.

<br>

<div id = "empty"></div>

### `bool` **empty**`() const`

Returns `true` if the matrix is empty, `false` otherwise.


<br>
<div id = "end_0"></div>

### `auto` **end**`() const`

Returns an iterator to the end of the data in the matrix.

<br>

<div id = "end_1"></div>

### `auto` **end**`() `

Returns an iterator to the end of the data in the matrix.

<br>

<div id = "end_column"></div>

### `virtual size_t` **end_column**`() const`

Provided for specifying were the columns end in the data.
 
Default implementation returns `_columns`.


<br>

<div id = "end_line"></div>

### `size_t` **end_line**`() const`

Alias for **[end_row](#end_row)**.

<br>

<div id = "end_row"></div>

### `virtual size_t` **end_row**`() const`

Provided for specifying were the rows end in the data.
 
Default implementation returns `_rows`.

<br>

<div id = "fill_0"></div>

### `Matrix&` **fill**`(const size_t &size , const T &value)`

Resizes the matrix to `(size ; size)` and fills it with `value`.

***WARNING:*** when T is dynamically allocated, ***MEMORY IS NOT FREED***.

<br>

<div id = "fill_1"></div>

### `Matrix&` **fill**`(const size_t &rows , const size_t &columns , const T &value)`

Resizes the matrix to `(rows ; columns)` and fills it with `value`.

***WARNING:*** when T is dynamically allocated, ***MEMORY IS NOT FREED***.


<br>

<div id = "for_each_0"></div>

### `Matrix&` `template<class Function>`**for_each**`(Function function)`

Applies `function` according to `std::for_each` to all the elements of the matrix.


<br>

<div id = "for_each_1"></div>

### `Matrix&` `template<class Function>`**for_each**`(Function function) const`

Applies `function` according to `std::for_each` to all the elements of the matrix.
 
This member function is the `const` equivalent of **[for_each](#for_each_0)**.


<br>

<div id = "hadamard"></div>

### `Matrix` **hadamard**`(const Matrix &arg) const`

Computes the Hadamard product (element-wise product) of the matrix and the argument.


<br>
<div id = "insert"></div>

### `Matrix&` **insert**`(const size_t &element_index , const Orientation &orientation , const std::vector<T> &data)`

Inserts `data` as either a row or a column (according to `orientation`) at the position `element_index`.


<br>
<div id = "insert_column"></div>

### `Matrix&` **insert_column**`(const size_t &index , const std::vector<T> &data)`

Inserts `data` as a column at the position `element_index`.

Calling this function is equivalent to:

```c++

    mat.insert(index , ste::Matrix<T>::Orientation::COLUMN , data);

```

<br>

<div id = "insert_line"></div>

### `Matrix&` **insert_line**`(const size_t &index , const std::vector<T> &data)`

Alias for **[insert_row](#insert_row)**.


<br>

<div id = "insert_row"></div>

### `Matrix&` **insert_row**`(const size_t &index , const std::vector<T> &data)`

Inserts `data` as a row at the position `element_index`.

Calling this function is equivalent to:

```c++

    mat.insert(index , ste::Matrix<T>::Orientation::ROW , data);

```

<br>
<div id = "inv"></div>

### `Matrix` **inv**`() const`

Returns the inverse of the matrix, as computed by **[operator!](#operator_not)**.

<br>
<div id = "invert"></div>

### `Matrix&` **invert**`()`

Inverts the matrix using **[operator!](#operator_not)**.

<br>
<div id = "isColumn"></div>

### `bool` **isColumn**`() const`

Returns `true` if the matrix is a column, `false` otherwise.

<br>
<div id = "isInvertible"></div>

### `bool` **isInvertible**`() const`

Returns `true` if the matrix is invertible, `false` otherwise.

<br>
<div id = "isLine"></div>

### `bool` **isLine**`() const`

Alias for **[isRow](#isRow)**

<br>
<div id = "isRow"></div>

### `bool` **isRow**`() const`

Returns `true` if the matrix is a row, `false` otherwise.

<br>
<div id = "isSquare"></div>

### `bool` **isSquare**`() const`

Returns `true` if the matrix is square, `false` otherwise.

<br>
<div id = "lineAt"></div>

### `std::vector<T>` **lineAt**`(const size_t &index) const`

Alias for **[rowAt](#rowAt)**.

<br>
<div id = "lines"></div>

### `const size_t&` **lines**`() const`

Alias for **[rows](#rows).**

<br>
<div id = "max"></div>

### `T` **max**`(std::function<T (const std::vector<T>&)> criterium = [](const std::vector<T> &data){return *std::max_element(data.begin() , data.end());}) const`

Returns the maximum of the matrix according to the argument.

Default uses `std::max_element`.

<br>
<div id = "mean"></div>

### `virtual T` **mean**`() const`

Alias for **[average](#average)**.

<br>
<div id = "min"></div>

### `T` **min**`(std::function<T (const std::vector<T>&)> criterium = [](const std::vector<T> &data){return *std::max_element(data.begin() , data.end());}) const`

Returns the minimum of the matrix according to the argument.

Default uses `std::min_element

<br>
<div id = "operator_equal_0"></div>

### `Matrix&` **operator=**`(const Matrix &arg)`

Assignement operator. Copies the contents of the argument in the matrix.

<br>
<div id = "operator_equal_1"></div>

### `Matrix&` **operator=**`(const std::vector<std::vector<T>> &arg)`

Assignement operator. Copies the contents of the argument in the matrix.
 
***Throws an exception*** if the size of all the vectors is not identical.

<br>
<div id = "operator_equal_2"></div>

### `Matrix&` **operator=**`(const std::vector<T> &arg)`

Assignement operator. Copies the contents of the argument in the matrix.
 
***Throws an exception*** if `arg.size() != _data.size()`, meaning you cannot resize a matrix with this operator.

<br>
<div id = "operator_mult_0"></div>

### `virtual Matrix` **operator\***`(const Matrix &arg) const`

Computes the usual matrix product of the object and the argument.
 
***Throws an exception*** if the sizes do not match.
 
A new object is created, meaning that a deep-copy occurs.


<br>
<div id = "operator_mult_1"></div>

### `Matrix` **operator\***`(const T &arg) const`

Computes the product of the matrix and a constant.
 
A new object is created, meaning that a deep-copy occurs.

<br>
<div id = "operator_mult_equal_0"></div>

### `Matrix&` **operator\*=**`(const Matrix &arg)`

Computes the usual matrix product of the object and the argument, and returns a reference to it.

<br>
<div id = "operator_mult_equal_1"></div>

### `Matrix&` **operator\*=**`(const T &arg)`

Computes the product of the matrix and a constant, and returns a reference to it.

<br>
<div id = "operator_plus_0"></div>

### `Matrix` **operator+**`(const Matrix &arg) const`

Computes the usual matrix sum of the object and the argument.
 
A new object is created, meaning that a deep-copy occurs.
 
***Throws an exception*** if the sizes do not match.

<br>

<div id = "operator_plus_1"></div>

### `Matrix` **operator+**`(const T &arg) const`

Adds the argument to all the elements of the matrix.
 
A new object is created, meaning that a deep-copy occurs.
 
<br>

<div id = "operator_plus_equal_0"></div>

### `Matrix&` **operator+=**`(const Matrix &arg)`

Computes the usual matrix sum of the object and the argument, and returns a reference to it.
 
***Throws an exception*** if the sizes do not match.

<br>
<div id = "operator_plus_equal_1"></div>

### `Matrix&` **operator+=**`(const T &arg)`

Adds the argument to all the elements of the matrix, and returns a reference to it.

<br>
<div id = "operator_minus_0"></div>

### `Matrix` **operator-**`(const Matrix &arg) const`

Computes the usual matrix difference of the object and the argument.
 
A new object is created, meaning that a deep-copy occurs.
 
***Throws an exception*** if the sizes do not match.


<br>
<div id = "operator_minus_1"></div>

### `Matrix` **operator-**`(const T &arg) const`

Substracts the argument to all the elements of the matrix.
 
A new object is created, meaning that a deep-copy occurs.


<br>
<div id = "operator_minus_2"></div>

### `Matrix` **operator-**`() const`

Returns the opposite of the matrix.
 
A new object is created, meaning that a deep-copy occurs.


<br>
<div id = "operator_minus_equal_0"></div>

### `Matrix&` **operator-=**`(const Matrix &arg)`

Computes the usual matrix sum of the object and the argument, and returns a reference to it.
 
***Throws an exception*** if the sizes do not match.



<br>
<div id = "operator_minus_equal_1"></div>

### `Matrix&` **operator-=**`(const T &arg)`

Substracts the argument to all the elements of the matrix, and returns a reference to it.


<br>
<div id = "operator_not"></div>

### `virtual Matrix` **operator!**`() const`

Computes the inverse of the matrix and returns it as a new object.
 
To fasten calculations, pre-computed formulas are used for matrices up to `(4 ; 4)`.

*If your matrices have specific properties that simplify the determination of the inverse, it is recommended to override this function in a subclass.*

**Explicitely computing an inverse is HEAVY. It is recommended to avoid such computations for matrices bigger than `(10 ; 10)`.**


<br>
<div id = "operator_power"></div>

### `virtual Matrix` **operator^**`(const long long int &arg) const`

Power operator. Returns the matrix to the specified power.
 
A new object is created, meaning that a deep-copy occurs.

***Throws an exception*** if the matrix is not square.
***Throws an exception*** if the matrix is not invertible and the power is negative.


<br>
<div id = "operator_power_equal"></div>

### `virtual Matrix&` **operator^=**`(const long long int &arg)`

Power operator. Returns the matrix to the specified power.

***Throws an exception*** if the matrix is not square.
***Throws an exception*** if the matrix is not invertible and the power is negative.




<br>
<div id = "operator_equality"></div>

### `virtual bool` **operator==**`(const Matrix &arg) const`

Equality operator.
 
Returns `true` if the argument and the matrix are of the same size and hold the same elements at the same positions.
 
*You may override this function if you need comparison by adresses.*
 


<br>
<div id = "operator_difference"></div>

### `virtual bool` **operator!=**`(const Matrix &arg) const`

Difference operator.
 
Returns the opposite of the result obtained by **[operator==](#operator_equality)**.


<br>
<div id = "print"></div>

### `virtual std::ostream&` **print**`(std::ostream &outstream = std::cout) const`

Prints the matrix to the speficied `std::outstream`, and returns a reference to the stream.
 
*You may override this function, should you prefer another format.*



<br>
<div id = "print_size"></div>

### `virtual std::ostream&` **print_size**`(std::ostream &outstream = std::cout) const`

Prints the size of matrix to the speficied `std::outstream`, and returns a reference to the stream.
 
*You may override this function, should you prefer another format.*


<br>
<div id = "push_back"></div>

### `Matrix&` **push_back**`(const std::vector<T> &data , const Orientation &orientation)`

Alias for **[add](#add).**


<br>
<div id = "push_back_column"></div>

### `Matrix&` **push_back_column**`(const std::vector<T> &data)`

Alias for **[add_column](#add_column).**


<br>
<div id = "push_back_line"></div>

### `Matrix&` **push_back_line**`(const std::vector<T> &data)`

Alias for **[add_line](#add_line).**


<br>
<div id = "push_back_row"></div>

### `Matrix&` **push_back_row**`(const std::vector<T> &data)`

Alias for **[add_row](#add_row).**


<br>
<div id = "remove"></div>

### `Matrix&` **remove**`(const size_t &element_index , const Orientation &orientation)`

Removes the row or column specified in argument, and returns a reference to the matrix.

***Throws an exception*** if the index is outside the matrix.


<br>
<div id = "remove_column"></div>

### `Matrix&` **remove_column**`(const size_t &index)`

Removes the column specified in argument, and returns a reference to the matrix.

***Throws an exception*** if the index is outside the matrix.



<br>
<div id = "remove_line"></div>

### `Matrix&` **remove_line**`(const size_t &index)`

Alias for **[remove_row](#remove_row)**.


<br>
<div id = "remove_row"></div>

### `Matrix&` **remove_row**`(const size_t &index)`

Removes the row specified in argument, and returns a reference to the matrix.

***Throws an exception*** if the index is outside the matrix.



<br>
<div id = "replace_0"></div>

### `Matrix&` **replace**`(const size_t &row , const unsigned &column , const T &value)`

Replaces the element at `(row ; column)` by `value` and returns a reference to the matrix.

***Throws an exception*** if the position is outside the matrix.


<br>
<div id = "replace_1"></div>

### `Matrix&` **replace**`(const size_t &index , const T &value)`

Replaces the element at the linear index in argument by `value` and returns a reference to the matrix.

***Throws an exception*** if the position is outside the matrix.



<br>
<div id = "replace_2"></div>

### `Matrix&` **replace**`(const size_t &value_index ,const Orientation &orientation , const std::vector<T> &value)`

Replaces the row or column at `value_index` by `value`, and returns a reference to the matrix.
 
***Throws an exception*** if the position is outside the matrix.
 
***Throws an exception*** if the size of `value` does not match the matrix' one.



<br>
<div id = "replace_3"></div>

### `Matrix&` **replace**`(const size_t &row_begin, const size_t &row_end ,const size_t &column_begin, const size_t &column_end,const T &value)`

Replaces the contents in the range specified by `value` and returns a reference to the matrix.
 
 ***Throws an exception*** if one of the positions is outside the matrix.


<br>
<div id = "replace_column"></div>

### `Matrix&` **replace_column**`(const size_t &value_index , const std::vector<T> &value)`

Replaces the column at `value_index` by `value` and returns a reference to the matrix.

***Throws an exception*** if the position is outside the matrix.
 
***Throws an exception*** if the size of `value` does not match the matrix' one.


<br>
<div id = "replace_line"></div>

### `Matrix&` **replace_line**`(const size_t &value_index , const std::vector<T> &value)`

Alias for **[replace_row](#replace_row)**.


<br>
<div id = "replace_row"></div>

### `Matrix&` **replace_row**`(const size_t &value_index , const std::vector<T> &value)`

Replaces the row at `value_index` by `value` and returns a reference to the matrix.

***Throws an exception*** if the position is outside the matrix.
 
***Throws an exception*** if the size of `value` does not match the matrix' one.


<br>
<div id = "reshape"></div>

### `Matrix&` **reshape**`(const size_t &rows , const size_t &columns)`

Changes the shape of the matrix to `(rows ; columns)`.

***Throws an exception*** if the number of elements is different.


<br>
<div id = "rowAt"></div>

### `std::vector<T>` **rowAt**`(const size_t &index) const`


Extracts a row of the matrix and returns it as a `std::vector<T>`.
 
***Throws an exception*** if the position is outside the matrix.


<br>
<div id = "rows"></div>

### `const size_t&` **rows**`() const`

Returns a reference to the total number of rows in the matrix.


<br>
<div id = "self_transpose"></div>

### `Matrix&` **self_transpose**`()`

***This function is still under developpment.***

Transposes the matrix in place and returns a reference to it.


<br>
<div id = "setDevice"></div>

### `Matrix&` **setDevice**`(const EXE &device)`

Changes the execution policy for the computations involving the matrix to `device`.


<br>
<div id = "size"></div>

### `const std::vector<size_t>` **size**`() const`

Returns a `std::vector<size_t>` holding `{_rows , _columns}`.


<br>
<div id = "sum"></div>

### `virtual T` **sum**`() const`

Computes the sum of all the elements of the matrix.


<br>
<div id = "swap"></div>

### `Matrix&` **swap**`(const size_t &element_1 , const size_t &element_2 , const Orientation &orientation)`

Swaps two rows or two columns at the specified indexes.
 
***Throws an exception*** if one of the positions is outside the matrix.

<br>
<div id = "swap_columns"></div>

### `Matrix&` **swap_columns**`(const size_t &element_1 , const size_t &element_2)`

Swaps the two columns at the positions in argument.
 
***Throws an exception*** if one of the positions is outside the matrix.


<br>
<div id = "swap_lines"></div>

### `Matrix&` **swap_lines**`(const size_t &element_1 , const size_t &element_2)`

Alias for **[swap_rows](#swap_rows)**.

<br>
<div id = "swap_rows"></div>

### `Matrix&` **swap_rows**`(const size_t &element_1 , const size_t &element_2)`

Swaps the two rows at the positions in argument.
 
***Throws an exception*** if one of the positions is outside the matrix.

<br>
<div id = "toVector1D_0"></div>

### `const std::vector<T>&` **toVector1D**`() const`

Returns a reference to the vector holding the data of the matrix.



<br>
<div id = "toVector1D_1"></div>

### `std::vector<T>&` **toVector1D**`()`

Returns a reference to the vector holding the data of the matrix.
 
***It is possible to modify the matrix through this method.***

<br>
<div id = "toVector2D"></div>

### `virtual std::vector<std::vector<T>>` **toVector2D**`() const`

Constructs a `std::vector<std::vector<T>>` from the matrix data.
 
**Data is ordered row by row.**

*This member is virtual, allowing you to change the export format.*


<br>
<div id = "trace"></div>

### `T` **trace**`() const`

Computes the trace of the matrix.



<br>
<div id = "transform"></div>

### `Matrix&` `template<class Function>` **transform**`(Function function)`

Applies `function` according to `std::transform` to all the elements of the matrix.


<br>
<div id = "transpose"></div>

### `Matrix` **transpose**`() const`

Returns the transpose of the matrix.


<br>
<div id = "transpose_in_place"></div>

### `Matrix&` **transpose_in_place**`()`

Alias for **[self_transpose](#self_transpose)**.



<br>
<br>



## Static member functions documentation

<br>
<div id = "eye"></div>

### `Matrix` **eye** `(const size_t &size , const EXE &device = EXE::CPU)`

Returns the identity matrix of dimensions `(size ; size)`.
 
Elements of the diagonal are `T(1)`.



<br>
<div id = "ones_0"></div>

### `Matrix` **ones** `(const size_t &size , const EXE &device = EXE::CPU)`

Returns a square matrix of dimensions `(size ; size)` filled with `T(1)`.



<br>
<div id = "ones_1"></div>

### `Matrix` **ones** `(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`

Returns a matrix of dimensions `(rows ; columns)` filled with `T(1)`.




<br>
<div id = "uniform_0"></div>

### `Matrix` **uniform** `(const size_t &size , const EXE &device = EXE::CPU)`

Returns a square matrix of dimensions `(size ; size)` filled with uniformly distributed numbers (according to `std::uniform_real_distribution<T>`).

<br>
<div id = "uniform_1"></div>

### `Matrix` **uniform** `(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`

Returns a matrix of dimensions `(rows ; columns)` filled with uniformly distributed numbers (according to `std::uniform_real_distribution<T>`).


<br>
<div id = "uniform_int_0"></div>

### `Matrix` **uniform_int** `(const size_t &size , const EXE &device = EXE::CPU)`

Returns a square matrix of dimensions `(size ; size)` filled with uniformly distributed numbers (according to `std::uniform_int<T>`).



<br>
<div id = "uniform_int_1"></div>

### `Matrix` **uniform_int** `(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`

Returns a matrix of dimensions `(rows ; columns)` filled with uniformly distributed numbers (according to `std::uniform_int<T>`).



<br>
<div id = "rand_0"></div>

### `Matrix` **rand** `(const size_t &size , const EXE &device = EXE::CPU)`

Alias for **[uniform](#uniform_0)**.

<br>
<div id = "rand_1"></div>

### `Matrix` **rand** `(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`

Alias for **[uniform](#uniform_1)**.





<br>
<div id = "randn_0"></div>

### `Matrix` **randn** `(const size_t &size , const T &mean = T(0) , const T &standard_deviation = T(1.), const EXE &device = EXE::CPU)`

Returns a square matrix of dimensions `(size ; size)` filled with normally distributed numbers (according to `std::normal_distribution<T>`) with the parameters in argument.



<br>
<div id = "randn_1"></div>

### `Matrix` **randn** `(const size_t &rows , const size_t &columns , const T &mean = T(0) , const T &standard_deviation = T(1) const EXE &device = EXE::CPU)`

Returns a matrix of dimensions `(rows ; columns)` filled with normally distributed numbers (according to `std::normal_distribution<T>`) with the parameters in argument.





<br>
<div id = "zeroes_0"></div>

### `Matrix` **zeroes** `(const size_t &size , const EXE &device = EXE::CPU)`

Returns a square matrix of dimensions `(size ; size)` filled with `T(0)`.


<br>
<div id = "zeroes_1"></div>

### `Matrix` **zeroes** `(const size_t &rows , const size_t &columns , const EXE &device = EXE::CPU)`

Returns a matrix of dimensions `(rows ; columns)` filled with `T(0)`.


<br>
<br>




## Non-member functions documentation

<br>
<div id = "element_wise_nm"></div>

### `Matrix` `template<class T>` **element_wise**`(const Matrix<T> &arg1 , const Matrix<T> &arg2)`

Alias for **[hadamard](#hadamard_nm)**.

Computes the Hadamard product (element-wise product) of the two matrices.


<br>
<div id = "for_each_nm_0"></div>

### `Matrix&` `template<class T , class Function>` **for_each**`(Matrix<T> &matrix , Function function)`

Applies `function` according to `std::for_each` to all the elements of the matrix in argument.


<br>
<div id = "for_each_nm_1"></div>

### `const Matrix&` `template<class T , class Function>` **for_each**`(const Matrix<T> &matrix , Function function)`

`const` equivalent of **[for_each](#for_each_nm_0)**.



<br>
<div id = "hadamard_nm"></div>

### `Matrix` `template<class T>` **hadamard**`(const Matrix<T> &arg1 , const Matrix<T> &arg2)`

Computes the Hadamard product (element-wise product) of the two matrices.



<br>
<div id = "invert_nm"></div>

### `Matrix&` `template<class T>` **invert**`(Matrix<T> &matrix)`

Inverts the matrix in argument and returns a reference to it.



<br>
<div id = "operator_print_mat_nm"></div>

### `std::ostream&` `template<class T>` **operator\<\<**`(std::ostream &outstream , const Matrix<T> &arg)`

Prints a matrix to a `std::ostream`, and returns a reference to the stream.



<br>
<div id = "operator_print_EXE_nm"></div>

### `std::ostream&` **operator\<\<**`(std::ostream &outstream , const EXE &a)`

Prints a `ste::EXE` to a `std::ostream`, and returns a reference to the stream.



<br>
<div id = "operator_binary_or_EXE_nm"></div>

### `EXE` **operator\|**`(const EXE &a , const EXE &b)`

Binary OR for `ste::EXE`.
 
Returns `EXE::GPU` if `a == EXE::GPU` or `b == EXE::GPU`, or `EXE::CPU` otherwise.


<br>
<div id = "operator_logical_or_EXE_nm"></div>

### `bool` **operator\|\|**`(const EXE &a , const EXE &b)`

Logical OR for `ste::EXE`.

Returns `true` if `a == EXE::GPU` or `b == EXE::GPU`.

<br>
<div id = "operator_binary_and_EXE_nm"></div>

### `bool` **operator&**`(const EXE &a , const EXE &b)`

Binary AND for `ste::EXE`.
 
Returns `EXE::GPU` if `a == EXE::GPU` and `b == EXE::GPU`, or `EXE::CPU` otherwise.


<br>
<div id = "operator_logical_and_EXE_nm"></div>

### `EXE` **operator&&**`(const EXE &a , const EXE &b)`

Logical AND for `ste::EXE`.
 
Returns `true` if `a == EXE::GPU` and `b == EXE::GPU`, or `false` otherwise.

<br>
<div id = "transform_nm"></div>

### `Matrix&` `template<class T , class Function>` **transform**`(Matrix<T> &matrix , Function function)`

Applies `function` to all the elements of `matrix`, according to `std::transform`.







## Non-member Types Documentation

<br>
<div id = "EXE"></div>

### `enum class` ste::EXE

`enum class` used to specify the execution policy for the calculations involving a ste::Matrix.

| Constant     | Value    | Description                                                                                                                                |
| -------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------|
|`EXE::CPU`    | 0        | All calculations involving this matrix will use the CPU, except the ones involving GPU matrices with `EXE::GPU` for `_device`.             |
|`EXE::C`      | EXE::CPU | Alias for EXE::CPU.                                                                                                                        |
|`EXE::HOST`   | EXE::CPU | Alias for EXE::CPU.                                                                                                                        |
|`EXE::GPU`    | 1        | All calculations involving this matrix will use the GPU. **This member only exists if** `STE_MATRIX_ALLOW_GPU` **has been** `#define`**d.**|
|`EXE::G`      | EXE::GPU | Alias for EXE::GPU.                                                                                                                        |
|`EXE::DEVICE` | EXE::GPU | Alias for EXE::GPU.                                                                                                                        |


<br>
<div id = "FMatrix"></div>

### `typedef Matrix<float>` **FMatrix**

Shortcut for `ste::Matrix<float>`.


<br>
<div id = "DMatrix"></div>

### `typedef Matrix<double>` **DMatrix**

Shortcut for `ste::Matrix<double>`.


<br>
<div id = "LDMatrix"></div>

### `typedef Matrix<long double>` **LDMatrix**

Shortcut for `ste::Matrix<long double>`.


<br>
<div id = "IMatrix"></div>

### `typedef Matrix<int>` **IMatrix**

Shortcut for `ste::Matrix<int>`.


<br>
<div id = "LIMatrix"></div>

### `typedef Matrix<long int>` **LIMatrix**

Shortcut for `ste::Matrix<long int>`.


<br>
<div id = "LLIMatrix"></div>

### `typedef Matrix<long long int>` **LLIMatrix**

Shortcut for `ste::Matrix<long long int>`.


<br>
<div id = "UIMatrix"></div>

### `typedef Matrix<unsigned int>` **UIMatrix**

Shortcut for `ste::Matrix<unsigned int>`.


<br>
<div id = "ULMatrix"></div>

### `typedef Matrix<unsigned long>` **ULMatrix**

Shortcut for `ste::Matrix<unsigned long>`.


<br>
<div id = "ULLMatrix"></div>

### `typedef Matrix<unsigned long long>` **ULLMatrix**

Shortcut for `ste::Matrix<unsigned long long>`.


<br>
<div id = "CMatrix"></div>

### `typedef Matrix<char>` **CMatrix**

Shortcut for `ste::Matrix<char>`.


<br>
<div id = "UCMatrix"></div>

### `typedef Matrix<unsigned char>` **UCMatrix**

Shortcut for `ste::Matrix<unsigned char>`.




<br>
<br>



## Macro Documentation

### STE_MATRIX_ALLOW_GPU

Enables the possibility to use the GPU for calculations.

When `STE_MATRIX_ALLOW_GPU` is defined, **[`ste::EXE`](#EXE)** gains the member `ste::EXE::GPU` and its aliases, allowing you to choose a device for the calculations.

See paragraph **[Using a GPU](#using-a-gpu)** for more information.


<br>
<br>
<br>

# License


This class is provided with the **GNU General Public License v3.0**.


See [ste-Matrix/LICENSE](https://github.com/Erellu/ste-Matrix/blob/master/LICENSE) for more information.




# Authors

***Developer / Tester:*** DUHAMEL Erwan (erwanduhamel@outlook.com)
 
***Tester:*** SOUDIER Jean (jean.soudier@insa-strasbourg.fr)
