import sys
import gc

a = "hello"

# sys.getrefcount(a)  # returns 2, one for the reference in the variable a and one for the reference
# in the argument of getrefcount
# gc.collect()  # forces garbage collection   

mylist = [1, 2, 3]
mylist.append(a)  

print("Reference count of a:", sys.getrefcount(a))  # returns 3, one for the reference in the variable a, one for the reference
# sys.getrefcount(a)  # returns 3, one for the reference in the variable a, one for the reference
# in the argument of getrefcount and one for the reference in the list mylist

print("Referrers of a:", gc.get_referrers(a))  # returns a list of objects that refer to a, which includes the list mylist

print("Collection thresholds:", gc.get_threshold())  # returns the current collection thresholds for the three generations of objects

gc.set_threshold(1000, 20, 30)  # sets the collection thresholds for the three generations of objects

print("Collection thresholds after setting:", gc.get_threshold())

print("Object counts:", gc.get_count())  # returns the current number of objects in each generation

gc.set_debug(False)  # enables debug output for the garbage collector