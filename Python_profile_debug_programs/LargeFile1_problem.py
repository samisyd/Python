'''
Absolutely. Here is a small, interview-friendly example that demonstrates the problem with reading
 a large file into memory and then fixes it using streaming/chunk processing
 Imagine we have a large CSV file containing customer transactions.

 What's the problem?
This line:
data = file.readlines()
loads the entire file into RAM.

For example:

large_file.csv = 5 GB

             ↓

       readlines()
             ↓
     
       RAM = 5+ GB

If your server only has 4 GB of available memory, the application could become extremely slow or even 
crash with an out-of-memory error.
'''

import os
# tracemalloc is a built-in Python module that helps you track memory allocations and identify
#  memory leaks.
import tracemalloc

def process_large_file(filename):
    # BAD: Loads the entire file into memory
    with open(filename, "r") as file:
        data = file.readlines()

    print(f"Number of lines: {len(data)}")


# Monitor memory
tracemalloc.start()

process_large_file("large_file.csv")

current, peak = tracemalloc.get_traced_memory()

print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory:    {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()