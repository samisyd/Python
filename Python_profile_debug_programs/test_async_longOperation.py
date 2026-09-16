
# https://www.youtube.com/watch?v=QlkXji08lno
# AsyncIO VS Threading VS Multiprocessing in Python 

import asyncio

async def long_operation():
    await asyncio.sleep(5)

async def main():
    try:
        await asyncio.wait_for(long_operation(), timeout=2)
    except asyncio.TimeoutError:
        print('Took too long...')

if __name__ == '__main__':
    asyncio.run(main())