import logging

'''
Here is how you can handle errors cleanly in generator pipelines and convert them to asynchronous streams for concurrent data handling.

1. Defensive Error Handling in Generator Pipelines

Because generator expressions evaluate lazily when iterated over, an unhandled exception inside a generator will halt the entire downstream processing chain. Wrapping individual stages inside custom generator functions with try/except keeps the pipeline resilient.
'''

# Simulated 1GB log stream (IP, Method, Path, Status Code, Bytes)
raw_log_data = """
192.168.1.10 - GET /index.html 200 1024
10.0.0.15 - POST /api/login 401 256
172.16.0.4 - GET /images/hero.png 200 20480
192.168.1.10 - GET /missing.html 404 512
10.0.0.22 - POST /api/checkout 500 128
"""

def safe_parse_log(log_stream):
    """Stage 2 with inline error handling."""
    for line in log_stream:
        try:
            parts = line.split()
            yield {
                "ip": parts[0],
                "method": parts[2],
                "path": parts[3],
                "status": int(parts[4]),
                "bytes": int(parts[5]),
            }
        except (IndexError, ValueError) as e:
            # Log the malformed record and keep the pipeline running
            logging.warning(f"Skipping malformed line '{line}': {e}")
            continue

# Pipeline remains clean and robust
lines = (line.strip() for line in raw_log_data.splitlines() if line.strip())
parsed_logs = safe_parse_log(lines)
error_logs = (log for log in parsed_logs if log["status"] >= 400)

# Consumption Stage: Processing the Stream
print("--- Error Stream Output ---")
for entry in error_logs:
    print(entry)