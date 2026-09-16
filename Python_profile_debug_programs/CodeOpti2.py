import io

# Simulated 1GB log stream (IP, Method, Path, Status Code, Bytes)
raw_log_data = """
192.168.1.10 - GET /index.html 200 1024
10.0.0.15 - POST /api/login 401 256
172.16.0.4 - GET /images/hero.png 200 20480
192.168.1.10 - GET /missing.html 404 512
10.0.0.22 - POST /api/checkout 500 128
"""

# Stage 1: Line Reader Generator (Lazy Reading)
# Yields raw lines one by one without loading the full file into memory
lines = (line.strip() for line in io.StringIO(raw_log_data.strip()) if line.strip())

# Stage 2: Parsing & Structuring Generator
# Splits each line into a dict structure
parsed_logs = (
    {
        "ip": parts[0],
        "method": parts[2],
        "path": parts[3],
        "status": int(parts[4]),
        "bytes": int(parts[5]),
    }
    for line in lines
    if (parts := line.split())
)

# Stage 3: Filtering Generator
# Filters only HTTP client and server errors (4xx and 5xx status codes)
error_logs = (log for log in parsed_logs if log["status"] >= 400)

# Stage 4: Formatting & Transformation Generator
# Extracts error details and adds a warning tag
formatted_errors = (
    f"[WARNING] {log['ip']} encountered {log['status']} on {log['path']} ({log['bytes']} bytes)"
    for log in error_logs
)

# Consumption Stage: Processing the Stream
print("--- Error Stream Output ---")
for entry in formatted_errors:
    print(entry)


'''
--- Error Stream Output ---
[WARNING] 10.0.0.15 encountered 401 on /api/login (256 bytes)
[WARNING] 192.168.1.10 encountered 404 on /missing.html (512 bytes)
[WARNING] 10.0.0.22 encountered 500 on /api/checkout (128 bytes)


Why This Pipeline Design Works Well

    Constant Memory Usage: Even if raw_log_data were 500 GB, Python never loads all lines or parsed dictionaries into memory simultaneously. Items are pulled through the pipeline one at a time via next() calls behind the scenes.

    Lazy Processing: Stage 4 requests an item, which pulls an item from Stage 3, which in turn pulls from Stage 2 and Stage 1. Processing stops immediately if you break out of the consuming loop early.

    Clean Decoupling: Each stage handles a single responsibility (reading, parsing, filtering, formatting), making it easy to swap or test individual components.
'''