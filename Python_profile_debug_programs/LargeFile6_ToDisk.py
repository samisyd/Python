'''
2. Stream Straight to SQLite Database

If you need to query or aggregate the results later, stream the filtered chunks 
directly into a lightweight SQLite database using Pandas.

| Approach | Peak Memory Usage | Best Used For | Code Pattern |
| :--- | :--- | :--- | :--- |
| **`results.append()` (In-Memory)** | Moderate (Grows with filtered output size) | Small filtered subsets that easily fit in RAM | `results.append(filtered)` then `pd.concat()` |

| **`to_csv(mode='a')` (Disk Append)** | Ultra Low (Stays constant at 1 chunk size) | Exporting filtered results directly to a flat file | `filtered.to_csv(file, mode='a', header=False)` |

| **`to_sql(if_exists='append')` (Database Stream)** | Ultra Low (Stays constant at 1 chunk size) | Storing output for indexing, SQL queries, or BI tools | `filtered.to_sql('table_name', conn, if_exists='append')` |


"I ran into a memory spike issue while processing large datasets with Pandas. Even though I was 
using chunksize to read the file in batches, I was appending the filtered chunks to a Python list 
and running pd.concat() at the end. As the filtered subset grew, memory consumption crept up and 
caused out-of-memory errors. I fixed this by streaming the chunks directly to disk using 
to_csv(mode='a') or a local database via to_sql(). This kept the memory footprint 
completely flat and constant, regardless of how large the input file or output subset was."

'''

import sqlite3
import pandas as pd

# Connect to (or create) a local SQLite database file
conn = sqlite3.connect("processed_data.db")

for chunk_df in pd.read_csv("large_file.csv", chunksize=10_000):
    filtered = chunk_df[chunk_df["amount"] > 1000]

    if not filtered.empty:
        # Append each chunk directly to a SQL table
        filtered.to_sql("high_value_transactions", conn, if_exists="append", index=False)

conn.close()
print("Done! Data streams directly into SQLite.")



'''

Here is a polished interview answer crafted to sound natural, technical, and high-impact. It builds
memory bottleneck while processing large CSV files. The existing code was appending filtered 
DataFrames into an in-memory list before doing a final pd.concat(). While chunking helped with the
 read step, accumulating all filtered records in RAM caused memory usage to scale linearly with 
 the dataset size.To solve this, I decoupled the processing from memory storage. I refactored the 
 pipeline to stream processed chunks directly to destination storage—appending to disk using 
 to_csv(mode='a') or streaming straight into a SQLite database with to_sql(if_exists='append'). 
 This transformed our memory profile from an growing linear curve to a completely flat, constant
 footprint equivalent to just a single chunk size."Key Talking Points (If Asked to Elaborate)
 The Core Flaw: results.append() delays disk write and holds state in RAM.The Solution: 
 Immediate write-out via file append (mode='a') or SQL batch inserting.
 
 The Impact: Memory 
 footprint drops to $O(1)$ constant space (tied to chunksize), eliminating 
 Out-Of-Memory (OOM) crashes regardless of whether the file is 1 GB or 100 GB.

 An interviewer will typically probe to see if you truly understand the trade-offs, edge cases, and modern alternatives rather than just memorizing a pattern.

Here are the most likely follow-up questions, what the interviewer is testing for, and how to answer them concisely.
1. "What happens if your pipeline fails halfway through writing to the CSV or SQLite table?"

    What they are testing: Resilience, idempotency, and transaction handling.

    How to answer:

        "For CSV appends, failure leaves a corrupted or partial file. To handle this, I write to a temporary file (output.csv.tmp) and atomically rename it only after the loop completes cleanly. For SQLite or SQL databases, I wrap the chunk processing inside a single database transaction (with conn:) so that if any chunk fails, the entire batch rolls back automatically."

2. "How did you determine the optimal chunksize?"

    What they are testing: Performance tuning and memory trade-offs.

    How to answer:

        "chunksize is a trade-off between I/O overhead and RAM usage. A tiny chunksize (like 100 rows) causes excessive I/O overhead from frequent disk writes. A huge chunksize (like 1,000,000 rows) risks spikey memory usage. I benchmarked memory using tracemalloc or psutil against different chunk sizes (e.g., 10k vs. 100k) to find the sweet spot where RAM stayed comfortably within system limits without bottlenecking I/O."

3. "If you need to sort, group by, or deduplicate across the whole dataset, chunking alone breaks down. How would you solve that?"

    What they are testing: System architecture limits of single-node Pandas chunking.

    How to answer:

        "For global operations like GROUP BY or DISTINCT, chunk-by-chunk processing alone isn't enough because data is distributed across chunks. If the dataset fits on a single machine, I would shift to DuckDB, which runs SQL queries directly on CSV/Parquet files out-of-core with low memory overhead. If the scale exceeds a single node, I’d transition the job to PySpark or Polars with lazy evaluation."

4. "Why stick with CSV for large file processing instead of Parquet or Feather?"

    What they are testing: File format knowledge and column-oriented storage.

    How to answer:

        "If I control the file output, I always prefer Parquet. CSVs require parsing text strings into data types on every read, which is slow and CPU-heavy. Parquet is a compressed, columnar format that supports predicate pushdown (reading only required columns and filtering metadata before loading rows into memory), drastically reducing I/O and parse time."

5. "How would you write unit tests for a chunked streaming pipeline?"

    What they are testing: Testing practices and maintainability.

    How to answer:

        "I decouple the transformation logic from the file streaming loop. I test the core filtering function on small Pandas DataFrames using pytest. For end-to-end pipeline testing, I mock the file input using Python’s io.StringIO with a small multi-chunk test file to verify that headers write correctly on chunk one and append seamlessly on chunk two."
'''