'''
Your current example is already a massive improvement over loading everything at once, but it still 
keeps all the filtered results in RAM. If the filtered subset is still large, memory usage 
will grow steadily throughout the loop.

Here are two better approaches, depending on where you want the final data to land:
1. The Direct Disk Appender (Zero-RAM accumulation)

Instead of appending chunks to a Python list (results.append()), write each chunk directly
to a CSV file. Memory usage stays flat, regardless of how many millions of rows you process.

1. The Direct Disk Appender (Zero-RAM accumulation)

Instead of appending chunks to a Python list (results.append()), write each chunk directly
to a CSV file. Memory usage stays flat, regardless of how many millions of rows you process.
'''

import pandas as pd

input_file = "large_file.csv"
output_file = "filtered_results.csv"

# Process the first chunk to create/overwrite the output file with headers
# Then append subsequent chunks without rewriting headers
first_chunk = True

for chunk_df in pd.read_csv(input_file, chunksize=10_000):
    # Process or filter the chunk
    filtered = chunk_df[chunk_df["amount"] > 1000]

    # Skip writing if the chunk has no matching rows
    if filtered.empty:
        continue

    # Write to disk immediately
    filtered.to_csv(
        output_file,
        mode="w" if first_chunk else "a",  # 'w' overwrites initially, 'a' appends after
        header=first_chunk,  # Header only on the first write
        index=False,
    )
    first_chunk = False

print(f"Done! Saved filtered results to {output_file}")