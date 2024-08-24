# Numpy revision notes

---

**Why NumPy?**

NumPy is optimized for performance and speed, as it is implemented in C. This makes operations on NumPy arrays faster compared to native Python lists. Additionally, NumPy's broadcasting feature allows for efficient operations without the need for manually rewriting loops.

```python
import numpy as np
print("Imported!")
```

**Data Structures in NumPy**

- **ndarray**: This is the core data structure in NumPy, representing n-dimensional arrays.

**Declaring ndarrays**

- **1-dimensional**

  ```python
  array1 = np.array([1, 2, 3])
  ```

- **2-dimensional**

  ```python
  array2 = np.array([
                      [1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]
                    ])
  ```

- **3-dimensional**

  ```python
  array3 = np.array([
                      [
                        [1, 2, 3], [4, 5, 6], [7, 8, 9]
                      ],
                      [
                        [1, 2, 3], [4, 5, 6], [7, 8, 9]
                      ],
                      [
                        [1, 2, 3], [4, 5, 6], [7, 8, 9]
                      ]
                    ])
  ```

**Printing and Type Checking**

```python
print(array1)
print(type(array1))

print(array2)
print(type(array2))

print(array3)
print(type(array3))
```

Output:
```
[1 2 3]
<class 'numpy.ndarray'>
[[1 2 3]
 [4 5 6]
 [7 8 9]]
<class 'numpy.ndarray'>
[[[1 2 3]
  [4 5 6]
  [7 8 9]]

 [[1 2 3]
  [4 5 6]
  [7 8 9]]

 [[1 2 3]
  [4 5 6]
  [7 8 9]]]
<class 'numpy.ndarray'>
```

**Array Shape**

- **Shape**: Defines the dimensions of the array.

```python
print(array1.shape)  # (3,)
print(type(array1.shape))

print(array2.shape)  # (3, 3)
print(type(array2.shape))

print(array3.shape)  # (3, 3, 3)
print(type(array3.shape))
```

Output:
```
(3,)
<class 'tuple'>
(3, 3)
<class 'tuple'>
(3, 3, 3)
<class 'tuple'>
```

**Array Dimensions**

- **ndim**: Number of dimensions.

```python
print(array1.ndim)  # 1
print(array2.ndim)  # 2
print(array3.ndim)  # 3

print(type(array1.ndim))
print(type(array2.ndim))
print(type(array3.ndim))
```

Output:
```
1
2
3
<class 'int'>
<class 'int'>
<class 'int'>
```

**Array Data Type**

```python
print(array1.dtype)  # int32
print(array2.dtype)  # int32
print(array3.dtype)  # int32
```

**Array Size**

- **size**: Total number of elements in the array.

```python
print(array1.size)  # 3
print(array2.size)  # 9
print(array3.size)  # 27
```

**Creating Arrays from Pandas DataFrame**

```python
import pandas as pd
print("Imported!")

df = pd.DataFrame(array2)
print(df)

print(df[0])  # 0th Column
print(df[1])  # 1st Column
print(df[2])  # 2nd Column
```

Output:
```
   0  1  2
0  1  2  3
1  4  5  6
2  7  8  9

0    1
1    4
2    7
Name: 0, dtype: int32
0    2
1    5
2    8
Name: 1, dtype: int32
0    3
1    6
2    9
Name: 2, dtype: int32
```

**Different Ways to Create Arrays**

- **Using `np.array` function**

  ```python
  normal_creation = np.array([1, 2, 3])
  print(normal_creation)
  ```

- **Using a Pandas Series**

  ```python
  series = pd.Series([1, 2, 3, 4])
  series_array = np.array(series)
  print(series_array)
  ```

- **Using a DataFrame**

  ```python
  df = {
    "rohit": [1, 2, 3],
    "amaya": [4, 5, 6],
    "kshitij": [7, 8, 9]
  }

  df_array = np.array(df)
  print(df_array)
  ```

**Special Arrays**

- **Array of Ones**

  ```python
  ones_with_dim_1 = np.ones((3,))
  ones_with_dim_2 = np.ones((2, 8))
  ones_with_dim_3 = np.ones((2, 4, 5))
  
  print(ones_with_dim_1)
  print(ones_with_dim_2)
  print(ones_with_dim_3)
  print(ones_with_dim_2.dtype)
  ```

- **Array of Zeros**

  ```python
  zeros = np.zeros((4, 4))
  print(zeros)
  print(zeros.dtype)
  ```

- **Empty Array**

  ```python
  empty_array = np.empty((3, 4))
  print(empty_array)
  ```

**Creating Arrays with Ranges**

```python
range_array = np.arange(0, 100, 5)
print(range_array)
print(type(range_array))
print(range_array.dtype)
```

**Creating Arrays with Random Numbers**

- **Random Integers**

  ```python
  random_array = np.random.randint(0, 10, size=(3, 5))
  print(random_array)
  print(type(random_array))
  print(random_array.dtype)
  ```

- **Random Floats**

  ```python
  random_arr = np.random.random((5, 5))
  print(random_arr)
  ```

- **Using Seed**

  ```python
  np.random.seed(9)
  random_array_seed = np.random.randint(0, 20, size=(5, 5))
  print(random_array_seed)
  ```

- **Unique Values**

  ```python
  unique_values = np.unique(random_array_seed)
  print(unique_values)
  ```

**Accessing Arrays and Elements**

- **Accessing Index in 1D Array**

  ```python
  print(array1[0])  # First element at 0th index

  for i in array1:
      print(i, end=" ")
  print()
  ```

---

**NumPy Array Operations and Manipulation**

**Importing Libraries**

```python
import numpy as np
import pandas as pd
print("Imported!")
```

**Generating Random Arrays**

```python
np.random.seed(0)
array1 = np.random.randint(0, 50, size=(4))
print(array1)  # [44 47  0  3]
```

**Creating Arrays of Ones**

```python
ones_array1 = np.ones((4))
print(ones_array1)  # [1. 1. 1. 1.]
```

**Arithmetic Operations**

- **Addition**

  ```python
  array1 + ones_array1
  # array([45., 48.,  1.,  4.])
  ```

- **Multiplication**

  ```python
  array1 * ones_array1
  # array([44., 47.,  0.,  3.])
  ```

- **Division**

  ```python
  array1 / ones_array1
  # array([44., 47.,  0.,  3.])
  ```

- **Floor Division**

  ```python
  array1 // ones_array1
  # array([44., 47.,  0.,  3.])
  ```

- **Subtraction**

  ```python
  array1 - ones_array1
  # array([43., 46., -1.,  2.])
  ```

- **Modulus**

  ```python
  array1 % ones_array1
  # array([0., 0., 0., 0.])
  ```

- **Exponentiation**

  ```python
  array1 ** 4
  # array([3748096, 4879681,       0,      81], dtype=int32)
  ```

- **Using `np.square` Method**

  ```python
  np.square(array1)
  # array([1936, 2209,    0,    9])
  ```

- **Square Root**

  ```python
  np.sqrt(array1)
  # array([6.63324958, 6.8556546 , 0.        , 1.73205081])
  ```

**Aggregation Functions**

- **Sum Comparison**

  ```python
  # Python list sum
  python_list = [i for i in range(1, 1000000)]
  %timeit sum(python_list)
  # 20.3 ms ± 477 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  # NumPy array sum
  numpy_array = np.array(range(1, 1000000))
  %timeit np.sum(numpy_array)
  # 279 µs ± 7.53 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
  ```

- **Mean**

  ```python
  print(array1)
  np.mean(array1)
  # 23.5
  ```

- **Maximum**

  ```python
  print(array1)
  np.max(array1)
  # 47
  ```

- **Minimum**

  ```python
  print(array1)
  np.min(array1)
  # 0
  ```

- **Median**

  ```python
  print(array1)
  np.median(array1)
  # 23.5
  ```

- **Standard Deviation**

  ```python
  print(array1)
  np.std(array1)
  # 22.051077071199945
  ```

- **Variance**

  ```python
  print(array1)
  np.var(array1)
  # 486.25
  ```

- **Standard Deviation and Variance Relationship**

  ```python
  print(np.std(array1))
  print(np.sqrt(np.var(array1)))  # Standard deviation = sqrt(variance)
  # 22.051077071199945
  ```

**Standard Deviation and Variance**

- **Standard Deviation**: Measures the spread of values from the mean.

  ```python
  print(np.std(array1))
  print(array1)  # Shows distance of every value from the std.

  print(np.mean(array1))
  # 23.5
  ```

- **Histogram Plotting**

  ```python
  %matplotlib inline
  import matplotlib.pyplot as plt
  print("Imported Matplotlib")
  
  plt.hist(array1)
  plt.show()
  ```

**Reshaping and Transposing Arrays**

- **Creating Arrays**

  ```python
  a1 = np.random.randint(1, 5, size=(2, 3))
  a2 = np.random.randint(1, 5, size=(2, 3, 3))
  print(a1, "\n")
  print(a2)
  ```

- **Reshaping Arrays**

  ```python
  dummy_arr1 = np.array([
      [1, 2, 3, 4],
      [5, 6, 7, 8]
  ])

  dummy_arr2 = np.array([1, 2, 3, 4])

  dummy_arr3 = np.array([
      [
          [1, 2, 3, 4],
          [5, 6, 7, 8]
      ],
      [
          [1, 2, 3, 4],
          [6, 7, 8, 9]
      ]
  ])

  print(np.shape(dummy_arr1))
  print(np.shape(dummy_arr2))
  print(np.shape(dummy_arr3))

  print(dummy_arr1.ndim)
  print(dummy_arr2.ndim)
  print(dummy_arr3.ndim)

  print(dummy_arr1.size)
  print(dummy_arr2.size)
  print(dummy_arr3.size)
  ```

  Reshaping `dummy_arr1`:

  ```python
  dummy_arr1.reshape((2, 4, 1))
  ```

  Multiplying arrays:

  ```python
  dummy_arr1 * dummy_arr3
  # array([[[ 1,  4,  9, 16],
  #         [25, 36, 49, 64]],
  #
  #        [[ 1,  4,  9, 16],
  #         [30, 42, 56, 72]]])
  ```

- **Transposing Arrays**

  ```python
  print(dummy_arr1.T)
  # array([[1, 5],
  #        [2, 6],
  #        [3, 7],
  #        [4, 8]])

  print(dummy_arr2.T)
  # array([1, 2, 3, 4])  # Transpose does not apply to 1D array

  print(dummy_arr3.T)
  # array([[[1, 1],
  #         [5, 6]],
  #
  #        [[2, 2],
  #         [6, 7]],
  #
  #        [[3, 3],
  #         [7, 8]],
  #
  #        [[4, 4],
  #         [8, 9]]])
  ```

**Dot Product and Element-Wise Multiplication**

- **Creating New Arrays**

  ```python
  np.random.seed(0)
  mat1 = np.random.randint(1, 10, size=(3, 5))
  mat2 = np.random.randint(1, 10, size=(3, 5))
  print(mat1, "\n\n", mat2)
  ```

- **Matrix Multiplication**

  ```python
  mat2 = mat2.T
  print(mat2)

  print(mat1.dot(mat2))
  # array([[148, 111, 137],
  #        [165, 115, 131],
  #        [255, 174, 149]])
  ```

---

**NumPy Array Comparison and Sorting**

**Importing Libraries**

```python
import numpy as np
print("Imported!")
```

**Comparison Operators**

Comparison operators return a boolean array where each element represents the result of the comparison between corresponding elements of the arrays.

- **Creating Random Matrices**

  ```python
  np.random.seed(0)
  mat1 = np.random.randint(1, 10, size=(5, 3))
  mat2 = np.random.randint(1, 10, size=(3, 5))

  print(mat1, "\n")
  print(mat2)
  # Output:
  # [[6 1 4]
  #  [4 8 4]
  #  [6 3 5]
  #  [8 7 9]
  #  [9 2 7]]
  # [[8 8 9 2 6]
  #  [9 5 4 1 4]
  #  [6 1 3 4 9]]
  ```

- **Comparison of Arrays**

  ```python
  # Error: shapes do not match
  # bool_array = (mat1 > mat2) 

  # Transpose mat2 to match shape with mat1
  mat2 = mat2.T
  bool_array = (mat1 > mat2)
  print(bool_array)
  # Output:
  # [[False False False]
  #  [False  True  True]
  #  [False False  True]
  #  [ True  True  True]
  #  [ True False False]]

  print(bool_array.dtype)
  # bool

  print(type(bool_array))
  # <class 'numpy.ndarray'>

  print(bool_array.size)
  # 15

  print(bool_array.ndim)
  # 2

  print(bool_array.shape)
  # (5, 3)
  ```

- **Comparison with Constant Values**

  ```python
  mat1 < 10
  # array([[ True,  True,  True],
  #        [ True,  True,  True],
  #        [ True,  True,  True],
  #        [ True,  True,  True],
  #        [ True,  True,  True]])

  mat2 >= 5
  # array([[ True,  True,  True],
  #        [ True,  True, False],
  #        [ True, False, False],
  #        [False, False, False],
  #        [ True, False,  True]])

  mat1 == mat1[3][2]
  # array([[False, False, False],
  #        [False, False, False],
  #        [False, False, False],
  #        [False, False,  True],
  #        [ True, False, False]])
  ```

**Sorting Arrays**

- **Sorting Arrays**

  Sorting returns a sorted copy of the array. For an in-place sort, use `np.sort`.

  ```python
  np.sort(mat1)
  # array([[1, 4, 6],
  #        [4, 4, 8],
  #        [3, 5, 6],
  #        [7, 8, 9],
  #        [2, 7, 9]])

  sorted_array = np.sort(mat2)  # Save sorted array for further use
  print(sorted_array)
  # Output (row-wise sorting):
  # [[6 8 9]
  #  [1 5 8]
  #  [3 4 9]
  #  [1 2 4]
  #  [4 6 9]]
  ```

- **Using `argsort`**

  `argsort` returns the indices that would sort the array.

  ```python
  argsorted_array = np.argsort(mat1)
  print(argsorted_array)
  # Output:
  # [[1 2 0]
  #  [0 2 1]
  #  [1 2 0]
  #  [1 0 2]
  #  [1 2 0]]

  # Applying sorting to individual rows
  np.sort(mat1[2])
  # array([3, 5, 6])
  ```

- **Finding Max and Min Indices**

  ```python
  np.argmax(mat1, axis=0)  # Indices of the max elements along each column
  # array([4, 1, 3], dtype=int64)

  np.argmin(mat1, axis=0)  # Indices of the min elements along each column
  # array([1, 0, 0], dtype=int64)
  ```

---

