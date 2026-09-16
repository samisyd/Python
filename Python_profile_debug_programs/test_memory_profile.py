# https://www.youtube.com/watch?v=3PdmLQIZpwE
# Memory Profiling in Python 

# https://www.youtube.com/watch?v=pVGujarYk9w
# Garbage Collection in Python: Speed Up Your Code

# https://www.youtube.com/watch?v=q_yk3oV14hE
#  Python AsyncIO Explained in 9 Minutes 

from memory_profiler import profile, memory_usage

log_file = open("memory_profile.log", "w+")

@profile(stream=log_file)
# @profile
def myfunction(list_size):
    print(f"Creating a list of size {list_size}...")
    mylist = ['hello'] * list_size
    mylist2 = ['world'] * list_size
    del mylist2
    return mylist

myfunction(10000000)


# mem_uage = memory_usage((myfunction, (), {'list_size': 100000}))
# mem_uage = memory_usage((myfunction, (), {'list_size': 10000000}), max_usage=True, interval=0.1)
# print(f"Memory usage: {mem_uage} MiB")