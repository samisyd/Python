'''
Optimizing Python Code

Performance Optimization Tips

    Use built-in functions and data structures when possible

    Avoid global variables in performance-critical code

    Use list comprehensions instead of loops for creating lists

    Consider using NumPy for numerical operations

    Profile your code to identify bottlenecks

Code Optimization Example

Less Efficient
Python
'''

# Creating a list of squares
def get_squares(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result

squares = get_squares(1000)

#  More Efficient

# Using list comprehension
def get_squares_optimized(n):
    return [i * i for i in range(n)]

# Or even better with generator
def get_squares_generator(n):
    return (i * i for i in range(n))


'''
List comprehensions run faster than traditional for loops in CPython mainly because they shift execution overhead from Python bytecode interpretation to underlying C code.

When creating a list using a standard loop and .append(), Python undergoes significant interpreter overhead on every single iteration:

    Method Lookups: On each iteration, Python must look up the .append attribute on the list object dynamically.

    Function Call Overhead: Calling .append() triggers CPython's full function evaluation stack, which involves pushing and popping frame objects for every element.

    Bytecode Instructions: A standard for loop executes multiple high-level bytecode instructions per iteration (e.g., FOR_ITER, STORE_FAST, LOAD_FAST, LOAD_ATTR, CALL_FUNCTION).

By contrast, a list comprehension uses specialized CPython bytecode instructions (BUILD_LIST and LIST_APPEND):

    Direct C-Level Execution: Instead of resolving and invoking the .append method in Python, LIST_APPEND calls the underlying C API function (PyList_Append) directly inside the C loop.

    Pre-allocated Capacity: CPython can often estimate or manage the list allocation space more efficiently in advance, reducing memory reallocations.

    Reduced Bytecode Loops: The loop logic is handled entirely at the C layer with minimal bytecode dispatches, avoiding repetitive variable and method lookups.
'''