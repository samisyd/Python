'''
2. ✅ Solution: Streaming

Python lets you read a file one line at a time.

Now the memory usage looks more like:

5 GB file
    ↓
Read 1 line
    ↓
Process line
    ↓
Release line
    ↓
Read next line

So you don't need 5 GB of RAM to process a 5 GB file.
'''
import tracemalloc

def process_large_file(filename):

    count = 0

    # GOOD: Read one line at a time
    with open(filename, "r") as file:

        for line in file:
            # Process the line
            count += 1

    print(f"Number of lines: {count}")


tracemalloc.start()

process_large_file("large_file.csv")

current, peak = tracemalloc.get_traced_memory()

print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory:    {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()