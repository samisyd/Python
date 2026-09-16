

import requests
import re
import time

import httpx
import asyncio


def count_https_in_web_pages():
    with open('websites.txt', 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f.readlines()]

    htmls = []
    for url in urls:
        htmls = htmls + [requests.get(url).text]

    count_https = 0
    count_http = 0
    for html in htmls:
        count_https += len(re.findall("https://", html))
        count_http += len(re.findall("http://", html))

    print('finished parsing')
    time.sleep(2.0)
    print(f'{count_https=}')
    print(f'{count_http=}')
    print(f'{count_https/count_http=}')

@profile
async def better_count_https_in_web_pages():
    with open('websites.txt', 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f.readlines()]

    # the response data can be accessed
        #  using the .text attribute of each response object. the structure of response 
        # data is the same as that of a normal httpx.Response object, so you can use the
        #  same methods and attributes to access the data. the structure is like text
        # and status_code, headers, etc.
    async with httpx.AsyncClient() as client:
        # this is a generator expression that creates a coroutine for each URL
        tasks = (client.get(url, follow_redirects=True) for url in urls)
        # gather runs all the coroutines concurrently and returns a list of responses
        # the reqs object will be a list of httpx.Response objects, one for each URL and will 
        # contain the response data for each request. 
        reqs = await asyncio.gather(*tasks)

    htmls = [req.text for req in reqs]

    count_https = 0
    count_http = 0
    for html in htmls:
        count_https += len(re.findall("https://", html))
        count_http += len(re.findall("http://", html))

    print('finished parsing')
    print(f'no of count_https: {count_https}')
    print(f'no of count_http: {count_http}')
    print(f'ratio: {count_https/count_http}')


def main():
    # import cProfile
    # import pstats

    start = time.perf_counter()

    # with cProfile.Profile() as pr:
    asyncio.run(better_count_https_in_web_pages())


    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:.2f} seconds")
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # # stats.print_stats()
    # stats.dump_stats(filename='needs_profiling.prof')


if __name__ == '__main__':
    main()