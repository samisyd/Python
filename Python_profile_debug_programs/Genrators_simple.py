import asyncio
import contextlib

'''

https://www.youtube.com/watch?v=N56Jrqc7SBk

Watch out for this (async) generator cleanup pitfall in Python

What happens to a try/finally, with block, or async with block inside a generator if the generator isn't exhausted? Does the cleanup code still run? When and how does it run? In this video we take a look at the answers to these questions and learn how to avoid a common situation where cleanup code doesn't run when you want it to.

# dont hold references to the generator object, and instead use a context manager to ensure that the generator is properly closed when it is no longer needed. This will ensure that any cleanup code in the generator is executed, even if the generator is not fully exhausted.
'''


class Resource:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"cleanup: {self.name}")


def gen():
    with Resource("database/lock/file/etc"):
        for x in range(3):
            print(f"yield {x}")
            yield x

def main():
    for x in gen():
        print(f"got {x}")
        if x == 1:
            break
    print("after loop")
