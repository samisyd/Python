import asyncio

async def backgroud_task():
    print('Running...')
    await asyncio.sleep(5)
    print('Finishing...')

async def main():
    task = asyncio.create_task(backgroud_task())

    print('Continuing immediately')

    await task

    print('But for this we need to wait')

if __name__ == '__main__':
    asyncio.run(main())
    print('This waits too')
