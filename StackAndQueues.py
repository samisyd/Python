#implement stack using Queue
# with pop expensive

import collections


class stackM:

    def __init__(self):
        self.q1 = collections.deque()
        self.q2 = collections.deque()
        self.curr_size = 0

    def push(self,val):
         
        self.q1.append(val)
        self.curr_size += 1

    def pop(self):

        if not self.curr_size:
            return -1
        while len(self.q1) > 1:
            popped = self.q1.popleft()
            self.q2.append(popped)
        
        retPopped = self.q1.popleft()
        self.curr_size -=1

        self.q1, self.q2 = self.q2, self.q1

        return retPopped

    def peek(self):
        return self.q1[self.curr_size - 1]

    def size(self):
        print(self.q1)
        return self.curr_size


    
mystack = stackM()

mystack.push(1)
mystack.push(5)
print(mystack.peek())
mystack.push(3)
print(mystack.pop())
mystack.push(2)

print(mystack.pop())
print("size is ", mystack.size())


arr = [2,3,1,2,4,3]
target = 7
'''
this solution does not work for negative values.  works only for positive values
arr2 = [2,7,3,-8,4,10]
target2 = 12
'''

# https://www.youtube.com/watch?v=aYqYMIqZx5s
def minSubArr(arr, target):

    l = 0
    minW = float("inf")
    totalSum = 0
    for r in range(len(arr)):
        totalSum += arr[r]
        while totalSum >= target:
            minW = min(minW, r-l+1)
            totalSum = totalSum - arr[l]
            l +=1

    return minW

print("start")
print(minSubArr(arr, target))

'''
# https://www.youtube.com/watch?v=gd9xEAnxXzc&list=PLEJXowNB4kPzEvxN8ed6T13Meet7HP3h0&index=6
Design special stack with getmin in O(1) time
use 2 stacks - 1. keep min in 1 stack
'''
# https://www.hackerrank.com/challenges/queue-using-two-stacks/problem
#Queue using Two Stacks
'''
STDIN   Function
-----   --------
10      q = 10 (number of queries)
1 42    1st query, enqueue 42
2       dequeue front element
1 14    enqueue 14
3       print the front element
1 28    enqueue 28
3       print the front element
1 60    enqueue 60
1 78    enqueue 78
2       dequeue front element
2       dequeue front element
'''
arr = [8, 10, 6, 3, 7]

class MyQueue:
    
    def __init__(self):
       self.st1 = []
       self.st2 = []
       
    def enqueue(self, val):        
        self.st1.append(val) 
    
    def dequeue(self):
        
        # if st1 is empty nothing to dequeue
        if not len(self.st1):
            return 
        
        while len(self.st1) > 1:
            self.st2.append(self.st1.pop())
        
        popped = self.st1.pop()
        while len(self.st2) > 0:
            self.st1.append(self.st2.pop())        
    
    def qprint(self):
        if len(self.st1):
            return self.st1[0]
        return 
    
mqueue =MyQueue()
#nQueries = input()
nQueries = 10
nQueries = int(nQueries)
for i in range(nQueries):
    
    query = input()
    val = query.split()
    # enqueue
    if val[0] == '1':        
        mqueue.enqueue(int(val[1]))
    elif val[0] == '2': #dequeue
        mqueue.dequeue()
    elif val[0] == '3': #print
        print(mqueue.qprint())
    else:
        pass



