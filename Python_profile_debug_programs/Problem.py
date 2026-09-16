'''
| Problem                  | Tool/technique               |
| ------------------------ | ---------------------------- |
| Memory leak              | `tracemalloc`                |
| CPU bottleneck           | `cProfile`                   |
| Line-by-line performance | `line_profiler`              |
| Memory consumption       | `memory_profiler`            |
| Large datasets           | Generators / iterators       |
| Large files              | Streaming / chunk processing |
| Database slowness        | Query profiling + indexes    |

'''


import tracemalloc
import time

def process_data():
    all_data = []

    for i in range(100):
        # Simulate a large dataset
        data = [f"Customer record {x}" for x in range(100_000)]

        # Problem: keeping every dataset in memory
        all_data.append(data)

        if i % 10 == 0:
            print(f"Processed batch {i}")

    return all_data


tracemalloc.start()

process_data()

current, peak = tracemalloc.get_traced_memory()

print(f"\nCurrent memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory:    {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()

print("Memory profiling complete.")

'''
What's wrong?

This line is the problem:

all_data.append(data)

Every batch is kept in all_data, so Python cannot release the memory.

As more data is processed:

Batch 1  → Memory increases
Batch 10 → Memory increases
Batch 50 → Memory increases
Batch 100 → Memory becomes very large
'''