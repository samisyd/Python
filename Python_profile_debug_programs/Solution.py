# 2. Fixed version using a generator

# Instead of keeping everything in memory, process one record at a time:
# this will keep memory usage low and stable. tracemalloc will show a much smaller peak
#  memory usage.
import tracemalloc

def generate_data():
    for i in range(100):
        for x in range(100_000):
            yield f"Customer record {x}"


def process_data():
    count = 0

    for record in generate_data():
        # Process the record
        count += 1
        print(record)

    return count


tracemalloc.start()

count = process_data()

current, peak = tracemalloc.get_traced_memory()

print(f"Records processed: {count}")
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory:    {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()

'''
The important difference is:

yield f"Customer record {x}"

Instead of creating and storing the entire dataset, the generator produces one item at a time.

Interview explanation

You can describe it like this:

"I had a Python application whose memory usage continuously increased while processing large datasets.
 I used tracemalloc to compare memory allocations and discovered that processed data was being 
 retained in a list. I replaced the list-based approach with generators so the application 
 could process records incrementally. This significantly reduced peak memory usage and 
 prevented memory growth."

Try this experiment

Run both versions and compare:

Memory problem → large and continuously growing memory
Generator      → much smaller and more stable memory

This is a great hands-on example for a Python performance/memory interview question.
'''