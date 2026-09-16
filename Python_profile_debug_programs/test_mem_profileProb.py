

import requests
import re
import time

# import httpx
# import asyncio
# from memory_profiler import profile, memory_usage
# log_file = open("memory_profile_ProfileSol.log", "w+")

@profile
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
    print(f'no of count_https: {count_https}')
    print(f'no of count_http: {count_http}')
    print(f'ratio: {count_https/count_http}')


def main():
    
    start = time.perf_counter()
    # import cProfile
    # import pstats

    # with cProfile.Profile() as pr:
    #     count_https_in_web_pages()

    # 1. Profile code execution
    # profiler = cProfile.Profile()
    # profiler.enable()

    count_https_in_web_pages()

    # profiler.disable()

    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # stats = pstats.Stats(profiler)
    # stats.strip_dirs()
    # # stats.sort_stats(pstats.SortKey.TIME)
    # stats.sort_stats('cumtime')   # Sort by cumulative time spent in function
    # stats.print_stats(10)         # Limit table output to top 10 rows
    # stats.print_stats()
    # stats.dump_stats(filename='needs_profiling.prof')


if __name__ == '__main__':
    main()