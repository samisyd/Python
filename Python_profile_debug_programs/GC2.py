import sys
import gc
import time

#  THIS IS IMPORTANT AND WILL SPEED UP YOUR APP MASSIVELY. 
# if you have a lot of db objects that causes a lot of references to be created and u dont have
# many unreachable objects, then u can disable GC or increase 
# the threshold and delay the GC. this will make the app run faster but the memory usage
#  will be higher.  
'''
gc.disable()  # disables automatic garbage collection
#  code
gc.enable()
# to enable manually and collect garbage, use gc.collect() to force a collection of all generations.
 gc.collect()  # forces garbage collection

'''

# gc.set_debug(True)  # enables debug output for the garbage collector
# gc.set_threshold(2000, 50, 100)  # sets the collection thresholds for the three generations 
# of objects
# dont do any garbage collection, just track the memory usage and leaks
# the code will run even faster..
# gc.disable() 

print("get current  threshhold", gc.get_threshold())  # returns the current collection thresholds for the three generations of objects

#  2. if u increase the threshhold and delay GC the app runs faster but the memory usage will be
#  higher. so u have to find a balance between memory usage and speed of the app.

# gc.set_threshold(3000, 50, 100)
# print("get current  threshhold", gc.get_threshold())

# 3. if u disable the GC and dont do any garbage collection, the app will run faster but the memory 
# usage will be higher.
#  gc.disable()

print("get current number of objects:", gc.get_count())  # returns the current number of objects in each generation
gc.collect(2) # forces garbage collection of generation 2 objects
print("get current number of objects:", gc.get_count())   # returns the current number of objects in each generation
gc.collect(0) # forces garbage collection of generation 0 objects

class Link:

    def __init__(self, next_link, data):
        self.data = data
        self.next = next_link

    def __repr__(self):
        return str(self.data)

l = Link(None, 'main link')

mylist = []

start = time.perf_counter()
for i in range(5000000):
    l_temp = Link(l, "l")
    mylist.append(l_temp)

end = time.perf_counter()

print("Time taken to create 100000 links:", end - start, "seconds")