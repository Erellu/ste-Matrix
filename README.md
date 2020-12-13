# ste::Matrix

C++ class that provides an interface for matrix-based computing

# Features

  •  Can  hold  any  class.
  •  Possibility  to  use  GPU  for  calculations  only by having `#define USE_GPU`[WIP]
  •  Fast  conversion  to  std::vector<T>  to  facilitate  GPU-acceleration-based  algorithms.

  

  •  Operators  *  and  +  available  as  long  as  the  template  parameters  provides  these  operators.
  •  Operator  -  avaible  for  any  type  that  accepts  (-1.)  as  parameter  in  its  constructor.

  

  •  Determinant,  inverse,  tranpose  ,cofactormatrix,  trace,  mean,  average.
  •  Classic  fill,  zeroes,  ones,  eye,  randn  and  rand.  
  •  Dynamic  resizing  (possibility  to  add,  remove  and  inverst  lines  and  /  or  columns)
  •  Fast  reshaping.

  •  Possibility  to  directly  print  contents  and  size  to  *stdout*.
  
  •  Possibility  to  override  most  functions  in  subclasses  to  increase  performances  in  specific  cases.


## Convinience types



|                |Type                           |Shortcut                   |
|----------------|-------------------------------|---------------------------|
|                |`Matrix<float>`           	 |`FMatrix`          		 |
|                |`Matrix<double>`           	 |`DMatrix`          		 |
|                |`Matrix<long double>`          |`LDMatrix`          		 |
|                |`Matrix<int>`           	 	 |`IMatrix`          		 |
|                |`Matrix<long int>`           	 |`LIMatrix`          		 |
|                |`Matrix<long long int>`        |`LLIMatrix`          		 |
|                |`Matrix<unsigned int>`         |`UIMatrix`          		 |
|                |`Matrix<unsigned long>`        |`ULMatrix`          		 |
|                |`Matrix<unsigned long long>`   |`ULLMatrix`          		 |
|                |`Matrix<char>`           	     |`CMatrix`          		 |
|                |`Matrix<unsigned char>`        |`UCMatrix`          		 |







# Member functions
>  Virtual  functions  are  marqued [v].
>  Static  functions  are  marqued  [S].


### Constructor:
  •  `Matrix`  |  Constructor.  Can accept  a  size  (x  ,  y  or  same  for  both) , a  `std::vector \<std::vector\<T\>\>`  or a `std::vector<T>` and a size to  construct  a  Matrix.

  
### Accessors:
  •  `size`  |  Returns  the  size  of  the  matrix,  as  `const  std::vector\<uint64_t\>`.

  • ` columns`  |  Returns  the  number  of  columns  of  the  matrix.

  •  `rows`  |  Returns  the  number  of  rows  of  the  matrix.

  • `lines`  |  Alias  for  `'rows'`.

  •  `elements`  |  Returns  the  total  number  of  elements  in  the  matrix.

  
### Information about the matrix shape:
  •  `isRow`  |  Returns  true  if  the  matrix  is  row,  false  otherwise.

  • `isLine`  |  Alias  for  `'isRow'`.

  •  `isColumn`  |  Returns  true  if  the  matrix  is  a  column,  false  otherwise.

  •  `isSquare`  |  Returns  true  if  the  matrix  is  square,  false  otherwise.

  •  `isInvertible`  |  Returns  true  if  the  matrix  is  invertible,  false  otherwise.

  •  `empty`  |  Returns  true  if  the  matrix  is  empty,  false  otherwise.

  
### Access to the matrix' contents:
  •  `at`  |  Returns  the  element  at  the  index  specified  in  argument.  It  is  passed  by  reference  when  the  matrix  is  non-const.  Linear  indexes  can  be  used.

  • `rowAt`  |  Returns  the  row at  the  specified  index.  It  is  passed  by  reference  when  the  matrix  is  non-const.

  •  `lineAt`  |  Alias  for  `'rowAt'`.

  •  `columnAt`  |  Returns  the  column  at  the  specified  index.  It  is  always  passed  by  value.

  
### Replacement:
***[v]***  •  `replace`  |  Replaces  the  element  specified  in  argument  first  (and  second  if  non-linear  mode  is  chosen)  by  the  value  in  last  argument.  **WARNING  !  If  T  is  dynamically  allocated,  memory  IS  NOT  freed.**

###  Appending to the matrix:

***[v]***  •  `add`  |  Adds  either  a  line  or  a  column  from  a  vector  at  the  end  of  the  matrix.

  •  `add_row`  |  Convience  function  to  add  a  row  at  the  end  of  the  matrix.

  •  `add_line` |  Alias  for  `'add_row'`.

  •  `add_column`  |  Convience  function  to  add  a  column  at  the  end  of  the  matrix.

  

  • `push_back`  |  Alias  for  `'add'`.

  •  `push_back_row`  |  Alias  for  `'add_row'`.

  • ` push_back_line`  |  Alias  for  `'add_line'`.

  •  `push_back_column`  |  Alias  for  `'add_column'`.

  
###  Removing from the matrix:
***[v]***   •  `remove`  |  Removes  either  a  line  or  a  column  at  the  position  specified.  **WARNING  !  If  T  is  dynamically  allocated,  memory  IS  NOT  freed.**

  •  `remove_row`  |  Convience  function  to  remove  a  row  at  a  specified  position. ** WARNING  !  If  T  is  dynamically  allocated,  memory  IS  NOT  freed.**

  •  `remove_line`  |  Alias  for  `'remove_row'.` ** WARNING  !  If  T  is  dynamically  allocated,  memory  IS  NOT  freed.**

  •  `remove_column`  |  Convience  function  to  remove  a  column  at  a  specified  position.  **WARNING  !  If  T  is  dynamically  allocated,  memory  IS  NOT  freed.**

  
###  Insertion:
***[v]***  •  `insert`  |  Inserts  either  a  line  or  a  column  at  the  position  specified.

  •  `insert_row`  |  Convience  function  to  insert  a  line  at  a  specified  position.

  • ` insert_line`  |  Alias  for  `'insert_row'`.

  • ` insert  column`  |  Convience  function  to  insert  a  column  at  a  specified  position.

  
###  Swaping elements:
***[v]***  •  `swap`  |  Swaps  two  lines  or  two  columns  at  the  positions  specified.

  •  `swap_row`  |  Convience  function  to  swap  two  rows  at  a  specified  positions.

  •  `swap_line`  |  Alias  for  `'swap_row'`.

  •  `swap_column`  |  Convience  function  to  swap  two  columns  at  a  specified  positions.

  
###  Reshaping:
  •  `reshape`  |  Changes  the  matrix  size  to  the  one  specified  in  argument.  Throws  an  exception  if  the  total  number  of  elements  in  the  new  size  does  not  match  the  current  one.

  
###  Converting the matrix to STL vectors:
***[v]***  •  `toVector2D`  |  Converts  the  matrix  to  `std::vector<std::vector<T>>`.

***[v]*** •  `toVector1D`  |  Converts  the  matrix  to ` std::vector<T>`

  
###  Printing the matrix:
***[v]***   •  `print`  |  Prints  the  contents  of  the  matrix  in  stdout.

***[v]***   •  `print_size`  |  Prints  the  size  of  the  matrix  in  stdout.

  
###  Iterator-like functions:
***[v]***   •  `begin_row`  |  Convinience  function  that  returns  0,  to  provide  syntax  as  close  as  the  one  relative  to  `std::algorithm`  as  possible.

  • ` begin_line`  |  Alias  for  `'begin_row'`.

***[v]***  •  `begin_column`  |  Convinience  function  that  returns  0,  to  provide  syntax  as  close  as  the  one  relative  to  `std::algorithm`  as  possible.

***[v]***  •  `end_row`  |  Convinience  function  that  returns  the  number  of  lines,  to  provide  syntax  as  close  as  the  one  relative  to  `std::algorithm`  as  possible

  • `end_line`  |  Alias  for  `'end_row'`.

***[v]***  •  `end_column`  |  Convinience  function  that  returns  the  number  of  columns,  to  provide  syntax  as  close  as  the  one  relative  to  `std::algorithm`  as  possible.

  

###  Sum, maximum, minimum, average:

***[v]***   •  `sum`  |  Returns  the  sum  of  all  elements  of  the  matrix,  as  `T`  (meaning  that  overflow  may  occur).

***[v]***  •  `mean`  |  Returns  the  mean  value  of  all  elements  of  the  matrix,  as  `T`  (meaning  that  rounding  error  and  overflow  may  occur).  It  is  computed  as  `sum()/(rows()*columns()`.

***[v]***   •  `average`  |  Alias  for `mean()`.

***[v]***  •  `max`  |  Returns  the  maximum  element  (according  to  `std::max_element`)  of  the  matrix.

***[v]***  •  `min`  |  Returns  the  minimum  element  (according  to  `std::min_element`)  of  the  matrix.

  
###  Matrix-algebra related functions:
***[v]***   •  `trace`  |  Returns  the  trace  of  the  matrix,  computed  as  `T`  (meaning  that  rounding  error  and  overflow  may  occur).  Throws  an  exception  (std::invalid_argument)  if  the  matrix  is  not  square.

***[v]***  •  `det`  |  Returns  the  determinant  of  the  matrix.  Throws  an  exception  (`std::invalid_argument`)  is  the  matrix  is  not  square.

***[v]***  •  `cofactor`  |  Returns  the  cofactor  of  the  specified  line  and  column  or  linear  index.  Throws  an  exception  if  one  of  them  is  outside  the  matrix.

***[v]***   •  `comatrix`  |  Returns  the  cofactor  matrix.  Convinience  function  that  returns  `cofactormatrix()`.

***[v]***   •  `cofactormatrix`  |  Returns  the  cofactor  matrix.

***[v]***   •  `transpose`  |  Returns  the  transpose  of  the  matrix.

  • `inv`  |  Returns  the  inverse  of  the  matrix  as  computed  by  operator!.

  •  `inverse`  |  Returns  the  inverse  of  the  matrix  as  computed  by  operator!.

**[S]**   •  `invert`  |  Returns  the  inverse  of  the  matrix  as  computed  by  operator!.

***[v]***  •  `hadamard`  |  Returns  the  Hadamard  product  of  two  matrices.  Throws  an  exception  if  the  sizes  do  not  match.

  •  `element_wise`  |  Convinience  function  that  returns  the  Hadamard  product  of  two  matrices. Calls `hadamard`.

  
###  Matrix creation:
***[v]***   •  `fill`  |  Resizes  the  matrix  as  specified  in  argument  and  fills  it  with  the  value  chosen.

**[S]**  •  `zeroes`  |  Resizes  the  matrix  as  specified  in  argument  and  fills  it  with  0.

**[S]**  •  `ones`  |  Resizes  the  matrix  as  specified  in  argument  and  fills  it  with  the  1.

**[S]**  •  `eye`  |  Creates  the  identity  matrix  of  the  size  specified  in  argument.

**[S]**   •  `randn`  |  Creates  a  matrix  of  normally  distributed  numbers.

**[S]**   •  `uniform`  |  Creates  a  matrix  of  uniformally  distributed  real  numbers.

**[S]**   • `uniform_int`  |  Creates  a  matrix  of  uniformally  distributed  integers.

**[S]**   •  `rand`  |  Alias  for  `'uniform'.`  Creates  a  matrix  of  uniformally  distributed  numbers.

  

  
###  Operators:
***[v]***   •  `operator=`  |  Assignment  operator.  Supports  assignments  from  std::vector<std::vector<T>>  and  from  other  Matrix.

***[v]***   •  `operator+`  |  Computes  the  addition  of  a  Matrix  with  another  (term  by  term).  Also  supports  the  addition  of  value  in  T.  In  that  case,  it  is  the  same  as  adding  a  Matrix  of  the  same  size  holding  only  the  same  value.

***[v]***  •  `operator*`  |  Computes  the  usual  matrix  product  of  two  matrices  or  multiplies  term  by  term  by  the  T  speficied.

***[v]***  •  `operator-`  |  Computes  the  substraction  of  two  Matrix.  Its  implementation  requires  T  to  be  able  to  be  multiplied  by  (-1.).

***[v]***  •  `operator!`  |  Returns  the  inverse  of  the  matrix.

***[v]***  •  `operator^` |  Returns  the  matrix  two  the  power  specifed  after  the  ^  (ex:  `a^2`  returns  `a*a`);

***[v]***   •  `operator==`  |  Equality  operator.  Returns  true  only  if  all  elements  are  identical  and  at  the  same  position.

***[v]***   •  `operator!=`  |  Returns  the  opposite  of  the  result  given  by `operator==`.


# Upcoming features:

- Determinant calculated on GPU.
- Cofactormatrix determined on GPU.
- Transpose determined on GPU.
- Invert determined on GPU.

# Authors

  DUHAMEL  Erwan  (erwanduhamel@outlook.com)
  SOUDIER  Jean  (jean.soudier@insa-strasbourg.fr)