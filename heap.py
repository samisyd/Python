# https://www.youtube.com/watch?v=pLIajuc31qk
class MinHeap:

    def __init__(self, arr=None):

        # build heap - takes O(n)
        self.heap = []
        if type(arr) == list and len(arr) > 0:
            self.heap = arr.copy()

            #takes O(n) time
            # picks the element in reverse order of array
            for i in range(len(self.heap))[::-1]:
                self.shiftdown(i)       

    def shiftup(self, i):
        
        parent = (i-1)//2
        while i!= 0 and self.heap[i] < self.heap[parent]:
            #swap
            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
            i = parent
            parent = (i-1)//2

    def shiftdown(self, i):

        leftchild = 2*i + 1
        rightchild = 2*i + 2

        while (leftchild < len(self.heap) and self.heap[leftchild] < self.heap[i]) or (rightchild < len(self.heap) and self.heap[rightchild] < self.heap[i]):
            #get min
            mini = leftchild if (rightchild >= len(self.heap)) or (self.heap[leftchild] < self.heap[rightchild]) else rightchild
            self.heap[mini], self.heap[i] = self.heap[i], self.heap[mini]
            i =  mini
            leftchild = 2*i + 1
            rightchild = 2*i + 2

    #O(logn)
    def insert(self, data):
        
        self.heap.append(data)
        self.shiftup(len(self.heap)-1)

    # O(1)
    def getmin(self):        
        return self.heap[0] if len(self.heap) > 0 else None

    # O(Log(n))
    def extractmin(self):
        
        if not len(self.heap): return None
        
        min = self.heap[0]
        #swap with last element        
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        self.heap.pop()
        self.shiftdown(0)
        return min

    def update_by_index(self, i, new):

        if not len(self.heap): return None

        old = self.heap[i]
        self.heap[i] = new

        if old < new:
            self.shiftdown(i)
        else:
            self.shiftup(i)

    def update(self, old, new):
        if not len(self.heap): return None

        if old in self.heap:
            self.update_by_index(self.heap.index(old), new)

    # given an array it returns the elements in assending order 
    # time complexity -  O(nlog(n)) -- extracting takes log(n) and doing for n elements 
    # selection sort - O(N^2) - because searching 1 takes O(n) and seaching n will take n^2  
    def heapsort(arr):
        #builds heap with new arr - takes O(n)
        heap = MinHeap(arr) 
        # extract min take logn
        return [heap.extractmin() for i in range(len(heap.heap))]


# used in Dijkstra, huffman, prims algo
class PriorityQueue:

    def init(self):
        self.queue = MinHeap()
    
    # O(log(n))
    def enqueue(self, element):
        self.queue.insert(element)

    # O(1)
    def peek(self):
        return  self.queue.getmin()
    
    #O(logn)
    def dequeue(self):
        return self.queue.extractmin()

    # O(logn)
    def change_priority_by_index(self, i, new):
        self.queue.update_by_index(i,new)

    # O(n)
    def change_priority(self, old, new):
        self.queue.update(old, new)

    def is_empty(self):
        return len(self.queue) == 0

arr = [9, 11, 18, 13, 15, 14, 7, 8, 12, 10, 4, 6, 3]
minh = MinHeap(arr)
print(minh.heap)

#for i in range(len(minh.heap)):
 #   print(minh.extractmin())

#GET KTH Largest element

import heapq

sarr = ["10", "6", "7", "3"]

# total time = n + k(2n) + n ==> 2n + k(2n) ==> 2kn
def getKthElement1(sarr, k):

    arri = [int(i) for i in sarr] # time n

    while k >1: 
        arri.remove(max(arri)) # time n + n (n - to find max and n to remove from arr)
        k -=1

    return str(max(arri)) # time n

print("getKthElement1 -->", getKthElement1(sarr, 3))

# total time - O(nlogn)
def getKthElement2(sarr, k):

    arri = [int(i) for i in sarr] # time n
    arri.sort() #nlog n
    return(arri[len(arri)-k])

print("getKthElement2 -->", getKthElement2(sarr, 3))


# *** IN PYTHON HEAPQ IS DESIGNED FOR MINHEAP
# totaltime - n + n + k-i(logn) + logn ==> 2n + klogn =>(n+klogn)
def getKthElement(sarr, k):

    # n time
    minHeapArr = [-(int(i)) for i in sarr]
    # building heap takes O(n)
    heapq.heapify(minHeapArr)

    # popping k-1 times will take (k-1(logn))
    while (k > 1):
        heapq.heappop(minHeapArr)
        k-=1
    
    # takes O(logn)
    return str(-(heapq.heappop(minHeapArr)))

print("getKthElement -->", getKthElement(sarr, 3))




'''
Jesse loves cookies and wants the sweetness of some cookies to be greater than value

. To do this, two cookies with the least sweetness are repeatedly mixed. This creates a special combined cookie with:

sweetness = 1 * Least sweet cookie + 2 * 2nd least sweet cookie).

This occurs until all the cookies have a sweetness >= k
Given the sweetness of a number of cookies, determine the minimum number of operations required. If it is not possible, return -1

https://www.hackerrank.com/challenges/one-week-preparation-kit-jesse-and-cookies

STDIN               Function
-----               --------
A[] size n = 6, k = 7
1 2 3 9 10 12       A = [1, 2, 3, 9, 10, 12]  

o/p = 2

Combine the first two cookies to create a cookie with sweetness = 1*1 + 2*2
After this operation, the cookies are .[5 3 9 10 12] 
Sort the cookies - [3 5 9 10 12]

2nd op = 3*1+5*2 = 13 and now it is [9 10 12 13]
all cookie now greater than k=7

return 2 - done in 2 ops
'''

def cookies(k, A):
    # Write your code here
    
       
    if len(A) <=1 or k == 0:
        return -1
    
    A.sort() # sort takes nlog(n)
    if A[0] >= k:
        return -1 
    
    def update(arr, res):
        
        global k
        if arr[0] >= k:
            return True
        
        val1 = arr.pop(0)
        val2 = arr.pop(0)
        total = val1 + (2*val2)
        print(val1, val2, total)
        arr.append(total)
        arr.sort() #sort takes nlog(n)
        print(arr)
        res[0] +=1
        update(arr, res)
    
    res = [0]
    retval = update(A, res)
    return res[0]

from heapq import heapify, heappush, heappop

#
# Complete the 'cookies' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER k
#  2. INTEGER_ARRAY A
#

def cookies2(k, A):
    # Write your code here    
       
    if len(A) <=1 or k == 0:
        return -1
    
    heapify(A) # takes O(n)
    res = 0
    
    while True:
        
        x = heappop(A) # takes logn
        if x >= k:
            return res
        
        if A:
            y = heappop(A) # takes logn)
            val = x + (y*2)
            heappush(A, val) # takes logn
            res+=1
        else:
            return -1

A = [1, 2, 3, 9, 10, 12] 
k = 7
print("cookies ==> ", cookies(k, A))
print("start")
A = [1, 2, 3, 9, 10, 12] 
print("cookies2 ==>", cookies2(k, A))



# "10-4+3*2+10/5"
def calc(strIn):

    op = '+'

    intarr = set(str(i) for i in range(10))
    operator = {'+','-','*','/'}
    mstack = []
    print(intarr)

    i = 0
    val = 0
    for chr in strIn:
        i+=1
        if chr in intarr:
            val = val*10 + int(chr)
            

        if chr in operator or i == len(strIn):
            
            if op == '+':
                mstack.append(val)
            elif op == '-':
                mstack.append(-val)
            elif op == '*':
                mstack[-1]  = mstack[-1] * val
            elif op == '/':
                mstack[-1]  = mstack[-1] // val
                
            op = chr
            val = 0
    print(sum(mstack))


print("printing calc")
calc("10-4+ 3*2 + 10/5")
