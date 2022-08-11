#import string

# https://www.youtube.com/watch?v=2EErQ25kKE8
#https://www.youtube.com/watch?v=zsJ-J08Qgdk
#create a basic calculator
# 1+2, 1-3, "10-4+3*2 + 10/5"
def calc(str1):

    curr, currop = 0, "+"
    stack = []
    operators = {'+', '-', '*', '/'}
    nums = set(str(i) for i in range(10))
    print(nums)
    i = 0
    for chr in str1:
        i+=1

        #print(currop,curr,chr)
        if chr in nums:
            curr = curr * 10 + int(chr)

        if chr in operators or i==len(str1):
            if currop == '+':
                stack.append(curr)
            if currop == '-':
                stack.append(-curr)
            if currop == '*':
                val = stack[-1] * curr
                stack[-1] = val
            if currop == '/':
                
                val = int(stack[-1] / curr)
                stack[-1] = val
            currop = chr
            curr = 0
            
            print(stack)

    return sum(stack)

print(calc("10-4+ 3*2 + 10/5"))


def calc2():
    pass

'''
Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.
Example 1:
Input: nums = [3,2,3]
Output: 3

Example 2:
Input: nums = [2,2,1,1,1,2,2]
Output: 2
https://www.youtube.com/watch?v=7pnhv842keE
'''

lis1 = [2,2,1,1,1,2,2]
list2 = [3,2,2] 
#using hashmap
def majEle(self, list1):

    count = {}
    res, maxCount = 0,0

    for n in list1:
        count[n] = 1 + count.get(n, 0)
        res = n if count[n] > maxCount else res
        maxCount = max(count[n], maxCount)  
    return res

#using O(1) space
# just keep the curr val and increment count of it as you see it. if u see diff element then keep decrementing
#  count until 0 then add the new element seen 
def majEle2(self, list1):

    res, count =0,0

    for n in list1:
        if count == 0:
            res = n
        count += (1 if n == res else -1)
    return res


'''
Squares of a sorted array
Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.
Example 1:

Input: nums = [-4,-1,2,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].

Example 2:

Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]

https://www.youtube.com/watch?v=FPCZsG_AkUg
'''

nums = [-7,-3,2,3,11]

def getsquares(nums):

    l,r = 0, len(nums)-1
    res =[0] * len(nums)
    
    i = len(nums) - 1
    while l <= r:
        lsq = nums[l] * nums[l]
        rsq = nums[r] * nums[r]
        if lsq > rsq:
            res[i] = lsq
            l+=1            
        else :
            res[i] = rsq
            r-=1
        i-=1
    
    return res

print(getsquares(nums))
print(nums[-1])
print(nums)
# prints the reverse of array
for i in range(len(nums))[::-1]:
    print(nums[i])


# Python code to sort a list of tuples 
# according to given key.
  
# get the last key.
def last(n):
    return n[m]  
   
# function to sort the tuple   
def sort(tuples):
  
    # We pass used defined function last
    # as a parameter. 
    return sorted(tuples, key = last)
   
# driver code  
a = [(23, 45, 20), (25, 44, 39), (89, 40, 23)]
m = 2
print("Sorted tuple:"),
#print(sort(a))
print(sorted(a, key = lambda x:x[1]))

# Python code to sort the lists using second element 
# of sublist Function to sort using sorted()
def Sort(sub_li):
  
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of 
    # sublist lambda has been used
    return(sorted(sub_li, key = lambda x: x[1]))    
  
# Driver Code
sub_li =[['rishav', 10], ['akash', 5], ['ram', 20], ['gaurav', 15]]
print(sorted(sub_li, key = lambda x: x[1]))
#print(Sort(sub_li))
'''
from datetime import datetime
listA = range(10000000)
setA = set(listA)
tupA = tuple(listA)

def calc(data, type):
    start = datetime.now()
    if data in type:
        print("")
        end = datetime.now()
    print(end-start)

calc(9999, listA)
calc(9999, tupA)
calc(9999, setA)
'''

arr = [1,2,3,4]
def prarr(arr):

    res = []
    for i in range(len(arr)):
        val = 1
        for j in range(len(arr)):
            if (i!=j):
                val *= arr[j]
        res.append(val)
    print(res)

prarr(arr)


#Product of all numbers
#multiplication of all except index
def prarr2(arr):

    n = len(arr)
    res = [1 for i in range(n)]
    arr1 = [1 for i in range(n)]
    arr2 = [1 for i in range(n)]
    
    for i in range(1,n):
        prod = arr[i-1] * arr1[i-1] 
        arr1[i] = prod

    for i in range(n-1)[::-1]:
        prod = arr[i+1] * arr2[i+1]
        arr2[i] = prod

    for i in range(n):
        res[i] = arr1[i] * arr2[i]

    print(res)

arr = [1,2,3,4]
prarr2(arr)

arr = ['a', 'b', 'c', 'd']
res = []
for i in arr:
    res.append(ord(i))
print(res)

def StrEd():
    
    noOps = input()
    print(int(noOps))
    strVal = ""
    strIn = []
    mystack = []
    noOps = int(noOps) 
    
    for i in range(noOps):
        strIn = input()
        print(strIn)
        val = strIn.split()
        print(val)
        op = int(val[0])
        print("mystack and strVal",mystack, strVal)

        if op == 1:
            mystack.append(strVal)            
            strVal+=val[1]
            print("in append",strVal)
            
        elif op == 2:
            mystack.append(strVal)
            value = int(val[1])
            if value <= len(strVal): 
                strVal = strVal[:-value]
            print("in del",strVal)
        elif op == 3:
            value = int(val[1])
            if value <= len(strVal):
                print(strVal[value-1])
            print("printing val")
        elif op == 4:
            print("in undo")
            strVal = mystack.pop()
        else:
            pass
        
StrEd()