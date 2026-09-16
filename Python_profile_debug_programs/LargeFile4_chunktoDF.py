'''
 how do you convert the chunked data to a df dataframe chunk by chunk

 Yes. With pandas, each chunk returned by pd.read_csv(..., chunksize=...) is already a DataFrame.

 CSV file
   ↓
pd.read_csv(chunksize=10_000)
   ↓
┌──────────────────────┐
│ chunk_df             │
│ pandas DataFrame     │
│ 10,000 rows          │
└──────────────────────┘
   ↓
process
   ↓
memory released
   ↓
┌──────────────────────┐
│ chunk_df             │
│ next 10,000 rows     │
└──────────────────────┘

If you want to combine the chunks into one DataFrame
You can, but then you lose the memory benefit:

chunks = pd.read_csv("large_file.csv", chunksize=10_000)
df = pd.concat(chunks, ignore_index=True)

print(df)

This eventually puts the entire dataset into memory.

Better approach for large files
Usually, you process each DataFrame and save only the result to a file or database, rather than 
keeping all chunks in memory.
This way, you only keep the processed results in memory, not the raw data.

chunk_df is already a DataFrame. You don't need to convert it.

If the file is 5 GB, for example, pandas might process it as:

5 GB CSV
   ↓
10,000 rows → DataFrame → process → discard
   ↓
10,000 rows → DataFrame → process → discard
   ↓
10,000 rows → DataFrame → process → discard
   ↓


'''

import pandas as pd

chunks = pd.read_csv("large_file.csv", chunksize=10_000)

for chunk_df in chunks:
    print(type(chunk_df))
    print(chunk_df.head())

    # Process this DataFrame
    total = chunk_df["amount"].sum()

    print("Chunk total:", total)

#  ------------------------------------------------------------------

# import pandas as pd

results = []

for chunk_df in pd.read_csv("large_file.csv", chunksize=10_000):

    # Each chunk_df is a DataFrame
    filtered = chunk_df[chunk_df["amount"] > 1000]

    # Keep only the small result
    results.append(filtered)

# Combine only the filtered results
# pd.concat will combine the filtered DataFrames into one final DataFrame
final_df = pd.concat(results, ignore_index=True)

print(final_df)

'''
Your current example is already a massive improvement over loading everything at once, but it still 
keeps all the filtered results in RAM. If the filtered subset is still large, memory usage 
will grow steadily throughout the loop.

Here are two better approaches, depending on where you want the final data to land:
1. The Direct Disk Appender (Zero-RAM accumulation)

Instead of appending chunks to a Python list (results.append()), write each chunk directly
to a CSV file. Memory usage stays flat, regardless of how many millions of rows you process.
'''