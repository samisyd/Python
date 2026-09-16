import time
import asyncio

async def io_task(name, delay, n_iter):
    for i in range(1, n_iter + 1):
        print(f"{name}: iteration {i}")
        await asyncio.sleep(delay)  # yield back to the event loop
        # time.sleep(delay)  # simulate some CPU-bound work

async def main():
    start = time.perf_counter()
    await asyncio.gather(
        io_task("Task A", 1.0, 3),
        io_task("Task B", 1.5, 3),
        io_task("Task C", 1.25, 3),
    )
    end = time.perf_counter()

    print(end - start)

    start = time.perf_counter()
    await io_task("Task A", 1.0, 3)
    await io_task("Task B", 1.5, 3)
    await io_task("Task C", 1.25, 3)
    end = time.perf_counter()

    print(end - start)

if __name__ == "__main__":
    asyncio.run(main())