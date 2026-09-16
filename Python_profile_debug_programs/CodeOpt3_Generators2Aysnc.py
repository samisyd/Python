'''
2. Asynchronous Generator Pipelines (async for / yield)

When input sources involve I/O bottlenecks (e.g., reading from network sockets, streaming from S3, or pulling from database cursors), async generators allow Python to release the event loop while waiting for data.

Key Differences: Synchronous vs. Asynchronous Generators

    Syntax: Async generators use async def and yield values inside async loops, consumed via async for or await anext().

    Concurrency: While standard generators pause full execution until the next .send() or next(), async generators yield execution back to the event loop during I/O waits, allowing other background tasks or requests to execute concurrently.

'''

import asyncio
from typing import AsyncGenerator

async def fetch_log_stream() -> AsyncGenerator[str, None]:
    """Simulates streaming log batches asynchronously from an external service."""
    mock_network_chunks = [
        "192.168.1.10 - GET /index.html 200 1024",
        "MALFORMED_LOG_LINE_HERE",  # Corrupted data
        "10.0.0.22 - POST /api/checkout 500 128",
    ]
    for line in mock_network_chunks:
        await asyncio.sleep(0.1)  # Simulate I/O latency
        yield line

async def async_filter_errors(stream: AsyncGenerator[str, None]) -> AsyncGenerator[dict, None]:
    """Asynchronously parses and filters high-status error events."""
    async for raw_line in stream:
        try:
            parts = raw_line.split()
            status = int(parts[4])
            if status >= 400:
                yield {"ip": parts[0], "status": status, "path": parts[3]}
        except (IndexError, ValueError):
            print(f"Skipping malformed line: {raw_line}")
            continue

async def main():
    log_stream = fetch_log_stream()
    error_stream = async_filter_errors(log_stream)

    # Consume stream as items arrive
    async for error in error_stream:
        print(f"Async Alert: {error['ip']} returned {error['status']} on {error['path']}")

# Run via asyncio event loop
asyncio.run(main())