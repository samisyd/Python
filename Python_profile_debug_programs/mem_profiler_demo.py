'''
pip install memory_profiler psutil

python -m memory_profiler memory_demo.py

Reading the Output

Output for load_file_all_at_once:
memory_profiler outputs a line-by-line report showing memory allocation (Mem usage) and how much
 memory each specific line added (Increment).

 Line #    Mem usage    Increment  Occurrences   Line Contents
============================================================
     5     48.2 MiB     48.2 MiB           1   @profile
     6                                         def load_file_all_at_once(filename):
     7     48.2 MiB      0.0 MiB           1       with open(filename, 'r') as f:
     8    850.5 MiB    802.3 MiB           1           lines = f.readlines()
     9    851.2 MiB      0.7 MiB     1000001           error_count = sum(1 for line in lines if 'ERROR' in line)
    10    851.2 MiB      0.0 MiB           1       return error_count


Notice line 8 spikes memory usage by +802.3 MiB instantly.
Output for stream_file_line_by_line

Line #    Mem usage    Increment  Occurrences   Line Contents
============================================================
    12     48.5 MiB     48.5 MiB           1   @profile
    13                                         def stream_file_line_by_line(filename):
    14     48.5 MiB      0.0 MiB           1       error_count = 0
    15     48.5 MiB      0.0 MiB           1       with open(filename, 'r') as f:
    16     48.6 MiB      0.1 MiB     1000001           for line in f:
    17     48.6 MiB      0.0 MiB     1000000               if 'ERROR' in line:
    18     48.6 MiB      0.0 MiB         500                   error_count += 1
    19     48.6 MiB      0.0 MiB           1       return error_count

    Memory consumption stays virtually flat at ~48.6 MiB total baseline throughout the loop.


'''

# memory_demo.py
from memory_profiler import profile
import pandas as pd

# 1. problematic approach: load whole file into memory
@profile
def load_file_all_at_once(filename):
    with open(filename, 'r') as f:
        # Load all lines into a list in RAM
        lines = f.readlines()
        error_count = sum(1 for line in lines if 'ERROR' in line)
    return error_count

# 2. Optimized approach: Stream line-by-line using generator-like iteration
@profile
def stream_file_line_by_line(filename):
    error_count = 0
    with open(filename, 'r') as f:
        # Open file object acts as an iterator, yielding 1 line at a time
        for line in f:
            if 'ERROR' in line:
                error_count += 1
    return error_count

# 3. Chunk processing using pandas
@profile
def process_file_in_chunks(filename, chunksize=10000):
    total_errors = 0
    # chunksize creates an iterator yielding DataFrames of size N
    for chunk in pd.read_csv(filename, chunksize=chunksize, names=['log_entry']):
        total_errors += chunk['log_entry'].str.contains('ERROR').sum()
    return total_errors

if __name__ == '__main__':
    filename = 'large_app_log.txt'
    
    # Run functions to compare profiles
    load_file_all_at_once(filename)
    stream_file_line_by_line(filename)
    process_file_in_chunks(filename)