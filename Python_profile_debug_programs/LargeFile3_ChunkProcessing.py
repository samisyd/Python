'''
3. ✅ Chunk processing

For CSV/data processing, you might want to process chunks rather than individual lines.

Using pandas:

Instead of:

df = pd.read_csv("large_file.csv")

which attempts to load everything:

5 GB CSV
   ↓
┌─────────────────────┐
│ Load entire file    │
│ into memory         │
└─────────────────────┘
          ↓
       💥 RAM

chunksize=10_000 does:

5 GB CSV
   ↓
┌──────────────┐
│ 10,000 rows  │ → process → release
└──────────────┘
        ↓
┌──────────────┐
│ 10,000 rows  │ → process → release
└──────────────┘
        ↓
┌──────────────┐
│ 10,000 rows  │ → process → release
└──────────────┘
        ↓
       ...

🧠 Interview answer

You could explain it like this:

"I had a memory issue while processing large files. The application was using readlines() to load the
 entire file into memory, which caused memory consumption to grow significantly. I identified 
 the issue using memory profiling and changed the implementation to stream the file line by 
 line. For larger data-processing jobs, I used chunk processing with pandas. This kept 
 memory usage much more stable because only a small portion of the file was 
 held in memory at a time."

'''

import pandas as pd

def process_large_file(filename):

    for chunk in pd.read_csv(filename, chunksize=10_000):

        # Process 10,000 rows at a time
        print(f"Processing {len(chunk)} rows")

        # Example processing
        total = chunk["amount"].sum()

        print("Chunk total:", total)


process_large_file("large_file.csv")