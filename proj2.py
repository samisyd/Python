import string


'''
# Python3 code to demonstrate 
# translations using 
# maketrans() and translate()
  
# specify to translate chars
str1 = "wy"
  
# specify to replace with
str2 = "gf"
  
# delete chars
str3 = "u"
  
# target string 
trg = "weeksyourweeks"
  
# using maketrans() to 
# construct translate
# table
table = trg.maketrans(str1, str2, str3)
  
# Printing original string 
print ("The string before translating is : ", end ="")
print (trg)
  
# using translate() to make translations.
print ("The string after translating is : ", end ="")
print (trg.translate(table))

'''

str1 = "listen to silent!''[;   "
str2 = "silent to listen"

def is_ana(str1, str2):

    #str1=str1.replace(string.punctuation,"")  -- doesnt work ... REplace just 1 char... if u need to replace muliple use re module (regular expressions)
    #str1=str1.replace(" ","")
    
    str1 = str1.translate(str1.maketrans('', '', string.whitespace + string.punctuation))
    print(str1)
    str2=str2.replace(" ","")

    if len(str1) != len(str2):
        return False


    alphab = "abcdefghijklmnopqrstuvwxyz"
    dict1 = dict.fromkeys(list(alphab), 0)
    dict2 = dict.fromkeys(list(alphab), 0)

    for i in range(len(str1)):
        dict1[str1[i]] += 1
        dict2[str2[i]] += 1

    return dict1 == dict2


print(is_ana(str1, str2))


L1 = [3 , 6, 1]
L2 = [4, 2, 0]
L3 = [2 , 7, 8]
A = [L1, L2, L3]
A1 = [L3]

def find_median(A):

    if len(A) == 1:
        vec = A[0]
        vec = sorted(vec)
        return  vec[len(vec)//2]
    else :
        mul_list = []
        for i in range(len(A)):           
            mul_list.extend(A[i])

        mul_list.sort()
        print(mul_list)
        return mul_list[len(mul_list) //2]

print("median is ", find_median(A))
print("median of A1 is ",find_median(A1))


s1 = 'waterbottle'
s2 = 'erbottlewat'


def is_rotation(str_1,str_2):

    if len(str_1) != len(str_2):
        return False

    if  str_1 == str_2:
        return True
    
    for _ in range(len(str_1)):

        str_1 = str_1[1:]+str_1[0]
        print(str_1)

        if str_1 == str_2:
            return True

    return False

def is_rot(str1, str2):

    return str2 in str1+str1 and len(str1) == len(str2)

print(is_rotation(s1,s2))
print(is_rot(s1,s2))


# The function is expected to return an INTEGER.
# find the element in the array using binary sort
# The function accepts following parameters:
#  1. INTEGER V
#  2. INTEGER_ARRAY arr
#  arr = [1, 4, 5, 7, 9, 12] V = 4
def introTutorial(V, arr):
    # Write your code here
    start = 0
    end = len(arr) -1
    mid = len(arr)//2
    
    while start <= end:
        mid = (start+end)//2
        if arr[mid] == V:
            return mid
        elif arr[mid] < V:
            start = mid+1
        else:
            end = mid-1

n = 4
for i in range(n-1,-1,-1):
    print(i)

for i in range(n, 0,-1):
    print(i)

def insertionSort(n, arr):
    # Write your code here
    
    lVal = arr[n-1]
    
    if lVal > arr[n-2]:
        print("")
        return
    for i in range(n-1,-1,-1):
        strs = ""
        if i != -1 and lVal > arr[i-1]:
            arr[i] = lVal
            strs = " ".join(map(str,arr))
            print(strs)
            break

        arr[i] =  arr[i-1]
        strs = " ".join(map(str,arr))
        print(strs)


#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
'''
Sample Input

5
2 4 6 8 3

Sample Output

2 4 6 8 8 
2 4 6 6 8 
2 4 4 6 8 
2 3 4 6 8 

'''

def insertionSort1(n, arr):
    # Write your code here
    
    key = arr[n-1]
    i = n-1
    
    while i > 0 and arr[i-1] > key:
        arr[i] = arr[i-1]
        print(*arr)
        i = i-1
    arr[i] = key
    print(*arr)

arr = [2, 4, 6, 8, 3]
print("calling insertion sort",insertionSort1(5, arr))
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
# sorts the whole arr

#https://www.hackerrank.com/challenges/insertionsort2/submissions/code/263174959
 

def insertionSort2(n, arr):
    for j in range(1, n):
        key = arr[j]
        i = j
        while i > 0 and arr[i-1] > key:
            arr[i] = arr[i-1]
            
            i -=1
        arr[i] = key
        print(*arr)
    

arr = [4,1,3,5,6,2]
arr2 = [2,4,6,8,3]
print("calling insertionsort2", insertionSort2(6, arr))

L1 = [3 , 6, 1]
L2 = [4, 2, 0]
L3 = [2 , 7, 8]
A = [L1, L2, L3]
A1 = [L3]

def median_matrix(A):
    new_list = []
    for row in A:
        new_list.extend(row)
        print(new_list)
    new_list.sort()
    print(new_list)
    if new_list:
        return new_list[len(new_list)//2]
    else:
        return False
    
print(median_matrix(A))

'''
The previous challenges covered Insertion Sort, which is a simple and intuitive sorting algorithm with a running time of n^2. 
In these next few challenges, we're covering a divide-and-conquer algorithm called Quicksort (also known as Partition Sort). 
This challenge is a modified version of the algorithm that only addresses partitioning. It is implemented as follows: 

STDIN       Function
-----       --------
5           arr[] size n =5
4 5 3 7 2   arr =[4, 5, 3, 7, 2]

Sample Output

3 2 4 5 7
Step 1: Divide
Choose some pivot element, p, and partition your unsorted array, arr, into three smaller arrays: left, right, and equal,
 where each element in left < p, each element in right > p , and each element in equal = p. 

'''
# O(nlogn) sorting time
def quickSort(arr):
    # Write your code here
    pivot = arr[0]
    left =[] 
    right = []
    equal = [pivot]
    for i in range(1, len(arr)):
        if arr[i] <= pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    
    print(left + equal + right)
    return left + equal + right

def fullquickSort(arr):
    # Write your code here
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left =[] 
    right = []
    equal = [pivot]
    for i in range(1, len(arr)):
        if arr[i] <= pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    
    print(left + equal + right)
    return fullquickSort(left) + equal + fullquickSort(right)


arr =[4, 5, 3, 7, 2]
print("starting quicksort ",quickSort(arr))
arr =[4, 5, 3, 7, 2, 9, 1]
print("starting fullquicksort2 ",fullquickSort(arr))


#
# Complete the 'countSort' function below.
#
# The function accepts 2D_STRING_ARRAY arr as parameter.
#

def countSort(arr):
    # Write your code here
    
    n = len(arr)
    res = [[] for i in range(100)]
    
    #first half of string should be -
    for i in range(n//2):
        res[int(arr[i][0])].append("-")
        
    #second half print values
    for i in range(n//2, n):
        res[int(arr[i][0])].append(arr[i][1])
    
    for item in res:
        if item:
            print(*item, end=' ')


arr = [[0, 'ab'],[6, 'cd'], [0, 'ef'], [6, 'gh'], [4, 'ij'], [0,'ab'], [6, 'cd'], [0, 'ef'], [6, 'gh'], [0, 'ij'],[4, 'that'], [3, 'be'], [0, 'to'],[1, 'be'],[5, 'question'], [1, 'or'],[2, 'not'], [4, 'is'], [2, 'to'],[4,'the']] 
countSort(arr)


#
# Complete the 'closestNumbers' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts INTEGER_ARRAY arr as parameter.
#
'''
Sorting is useful as the first step in many different tasks. The most common task is to make finding
 things easier, but there are other uses as well. In this case, it will make it easier to determine which 
 pair or pairs of elements have the smallest absolute difference between them.

Note
As shown in the example, pairs may overlap.
Given a list of unsorted integers, arr , find the pair of elements that have the smallest absolute 
difference between them. If there are multiple pairs, find them all.

Function Description
Complete the closestNumbers function in the editor below.
closestNumbers has the following parameter(s):
    int arr[n]: an array of integers
Returns
- int[]: an array of integers as described 

Sample Input 2
4
5 4 3 2

Sample Output 2
2 3 3 4 4 5

Explanation 2
Here, the minimum difference is 1. Valid pairs are (2, 3), (3, 4), and (4, 5). 
'''

def closestNumbers(arr):
    # Write your code here
    arr.sort()    
    mindiff = float('inf')
    res = []
    
    for i in range(1, len(arr)):
        newdiff = abs(arr[i] - arr[i-1])
        if  newdiff < mindiff:
            res = [arr[i-1], arr[i]]
            mindiff = newdiff
        elif newdiff == mindiff:
            res.extend([arr[i-1], arr[i]])
    
    return res

arr = [5,4,3,2]

print("...")
print("...")
print("closest num",closestNumbers(arr))
print("...")
print("...")


arr1 = [2,3,5,6,7,10,0,0,0,0,0]
arr2 = [1,4,6,8,9]



def Merge2SortedArrays(arr1, m, arr2, n):

    
    last = m+n-1
    i = m-1
    j = n-1
    while i >= 0 and j >=0:

        if arr1[i] >= arr2[j]:
            arr1[last] = arr1[i]
            i-=1
        else:
            arr1[last] = arr2[j]
            j-=1
        last-=1
    
    print(i,j)
    while j >= 0:
        arr1[last] = arr2[j]
        last, j = last-1,j-1
    return arr1



print(Merge2SortedArrays(arr1, 6, arr2, 5))


# Good pair problem
'''
given arr of int create a func that returns the no of good pairs , where a pair (i,j) is called 
good if i<j and arr[i] == arr[j]
# Good pair problem
# https://www.youtube.com/watch?v=qnCKQQE6wLM
'''
goodPair = [4, 2, 1, 3, 5, 1, 3, 2, 6]

# brute force - order n^2
def findgoodpair2(arr):
    count = 0
    for i in range(0, len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                count +=1
    return count

print("findgoodpair2", findgoodpair2(goodPair))

# order is n and space complexity is n

def findgoodpair(arr):
    pair = 0
    res = []
    dict = {}
    n = len(arr)
    for i in range(n):        
        if arr[i] in dict.keys():            
            dict[arr[i]] += 1
            pair +=1
        else:
            dict[arr[i]] = 1
    print(dict)
    return pair

print(findgoodpair(goodPair))

'''
your company built an in-house calendar tool called HiCal. You want to add a feature to see the
 times in a day when everyone is available. To do this, you’ll need to know when any team is 
 having a meeting. In HiCal, a meeting is stored as a tuple ↴ of integers (start_time, end_time). 
 These integers represent the number of 30-minute blocks past 9:00am.

For example:
(2, 3)  # Meeting from 10:00 to 10:30 am
(6, 9)  # Meeting from 12:00 to 1:30 pm

Write a function merge_ranges() that takes a list of multiple meeting time ranges and returns a list of condensed ranges.

For example, given:

  [(0, 1), (3, 5), (4, 8), (10, 12), (9, 10)]
  [(0, 1), (3, 5), (4, 8), (9, 10), (10, 12)]

your function would return:

  [(0, 1), (3, 8), (9, 12)]

https://www.interviewcake.com/question/python/merging-ranges?course=fc1&section=array-and-string-manipulation
'''


arr = [(0,1),(3,5),(4,8),(10,12),(9,10)]

def MeetingRoom(arr):

    arr.sort()
    print(arr)
    startTime, endTime = arr[0]
    print(startTime, endTime)

    res = []
    for i in range(1, len(arr)):

        nxtSTime, nxtETime = arr[i]
        if nxtSTime > endTime:
            res.append((startTime,endTime))
            startTime = nxtSTime
            endTime =  nxtETime
        else:
            if endTime < nxtETime:
                endTime = nxtETime
        
    res.append((startTime,endTime))
    return res

print(MeetingRoom(arr))


# Reverse String - 3 Ways - Leetcode 344
# 1. using 2 pointers and in place, 2. using stack 3. using recursion
# STRINGS are Immutable - YOU CANNOT change/replace the characters so a character array is given 
def reverseStr(str):

    l,r = 0, len(str)-1

    while l<r:
        str[l],str[r] = str[r], str[l]
        l , r = l+1, r-1

# using stack
def reverseStr2(str):

    stack = []
    for i in str:
        stack.append(i)
    
    i = 0
    while stack:
        str[i] = stack.pop()
        i+=1

#using recursion
def reverseStr3(str):

    def revstrrec(l,r):
        if l<r:
            str[l], str[r] = str[r], str[l]
            revstrrec(l+1, r-1)

    revstrrec(0,len(str)-1)

strd = ['s','a','m', 'i']
reverseStr2(strd)
print(strd)

#https://www.youtube.com/watch?v=YPTqKIgVk-k
#Top K Frequent Elements - Bucket Sort - Leetcode 347 - Python
'''
This can be solved in 3 ways
1. list => store in a arraylist with [value,count] and then sort based on counts - will take nlogn time
2. heapify => use maxheap and add each pair to maxheap and key will be based on counts/ocuurances of the value and pop exactly k times.
heapify will take time n to add all elements and popping k elements will take klogn time
3. bucketsort ==> we create an empty arrayList of same size as input. Create hashmap to get count of each element.
 and then place the hashmap values in empty arrayList dpending on the count which is the index of the arrayList     
'''

#time taken - n+nlogn , space = n
arr = [1,1,2,2,3]
k = 2
def topkElements(arr, k):

    hmap = {}
    for i in arr:
        hmap[i] = 1+ hmap.get(i, 0)
    
    res = []
    i = 0
    
    for key, value in sorted(hmap.items(), key=lambda x: x[1], reverse= True ):
        res.append(key) 
        i +=1
        if i == k:
            return res

print("using hash",topkElements(arr, k))

import heapq
# *** IN PYTHON HEAPQ IS DESIGNED FOR MINHEAP
arr = [1,1,1,2,2,3,3,3,4]
k = 4
def topkElementsheap(arr, k):

    hmap = {}
    for i in arr:
        hmap[i] = 1+ hmap.get(i, 0)
    
    res = []
    arrlist = []
    for key,value in hmap.items():
        arrlist.append([-value,key])
    print(arrlist)
    # takes time = n
    heapq.heapify(arrlist)
    print(arrlist)
    #klogn
    for i in range(k):
        val,key = heapq.heappop(arrlist) # takes logn
        res.append(key)
    
    return res

print("topkElementsheap values",topkElementsheap(arr, k))


# time taken - n+n+n and space comp = n+n+n
k = 3
arr =[1,1,1,2,2,2,3,3]
# 3. BCKET sort method
def topKfrequentEle(arr, k):

    arrlist = [[] for n in range(len(arr) +1)]
    arrh = {}
    for i in arr:
        arrh[i] = 1 + arrh.get(i,0)

    for item,count in arrh.items():
        arrlist[count].append(item)
    print(arrlist)

    res = []
    for m in range(len(arrlist))[::-1]:
        for n in arrlist[m]:
            res.append(n)
            if len(res) == k:
                return res

print(topKfrequentEle(arr, k))


orders = {
	'cappuccino': 54,
	'latte': 56,
	'espresso': 72,
	'americano': 48,
	'cortado': 41
}
print("1")
sort_orders = sorted(orders.items(), key=lambda x: x[1], reverse=True)
print("2")
#sort_orders = sorted(orders.values())

for i in sort_orders:
	print(i[0], i[1])

# return type is list
print(type(sort_orders))

'''
Initially, reverse the individual words of the given string one by one, 
for the above example, after reversing individual words the string 
should be “i ekil siht margorp yrev hcum”.
Reverse the whole string from start to end to get the desired output
“much very program this like i” in the above example.
'''
# Python3 program to reverse a string
 
# Function to reverse each word in the string
def reverse_word(s, start, end):
    while start < end:
        s[start], s[end] = s[end], s[start]
        start = start + 1
        end -= 1
 
 
s = "i like this program very much"
 
# Convert string to list to use it as a char array
s = list(s)
print("printing s",s)
start = 0
while True:
     
    # We use a try catch block because for
    # the last word the list.index() function
    # returns a ValueError as it cannot find
    # a space in the list
    try:
        # Find the next space
        end = s.index(' ', start)
        print(end)
 
        # Call reverse_word function
        # to reverse each word
        reverse_word(s, start, end - 1)
        #Update start variable
        start = end + 1
 
    except ValueError: 
        # Reverse the last word
        reverse_word(s, start, len(s) - 1)
        break
 
# Reverse the entire list
s.reverse()
 
# Convert the list back to
# string using string.join() function
s = "".join(s)
 
print(s)



'''
Write a function reverse_words() that takes a message as a list of characters and 
reverses the order of the words in place. 

Why a list of characters instead of a string?

The goal of this question is to practice manipulating strings in place. 
Since we're modifying the message, we need a MUTABLE type like a list, 
instead of Python 2.7's IMMUTABLE STRINGS. 


    
'''
message = [ 'c', 'a', 'k', 'e', ' ',
            'p', 'o', 'u', 'n', 'd', ' ',
            's', 't', 'e', 'a', 'l' ]

def reverse_words2(message):

    i = 0
    n = len(message)
    j = n-1

    # reverse all the chars
    while i<j:

        message[i], message[j] = message[j], message[i]
        i +=1
        j-=1
    print(message)

    i = j = k = 0
    
    # wal through the while string
    while k < n:
        print(k)
        # get  each word
        while k < n and message[k] != " " :
            k+=1
        j = k-1
        print(k)
        
        #swap the letters of the words
        while i < j:
            #swap i and j
            message[i], message[j] = message[j],message[i]
            i +=1
            j -=1
        
        print(message)
        k +=1
        i = j = k
        print(i,j,k)

    print(message)

reverse_words2(message)


# this method makes a string and splits it and then reverses it
message = [ 'c', 'a', 'k', 'e', ' ',
            'p', 'o', 'u', 'n', 'd', ' ',
            's', 't', 'e', 'a', 'l' ]

def reverse_words(message):

    stri = "".join(message)
    stri = stri.split()
    n = len(stri)
    
    i = 0
    print(stri)
    # check if even strings
    if n % 2 == 0:

        j = n // 2
        while j < n:
            # keep swpapping
            stri[j], stri[n-j-1] =  stri[n-j-1], stri[j]
            j+=1 
        
    else: # else odd no of strings
        j = (n // 2) + 1
        while j < n:
            # keep swpapping
            stri[j], stri[n-j-1] =  stri[n-j-1], stri[j]
            j+=1
    
    res = []
    print(stri)
    for item in stri:
        res.extend(list(item))
        res.extend(" ")
    print(res)
 
# Prints: 'steal pound cake'
#print(''.join(message))

reverse_words(message)

'''
 Write a function that takes:

    a list of unsorted_scores
    the highest_possible_score in the game

and returns a sorted list of scores in less than O(nlogn) time. 
'''
unsorted_scores = [37, 89, 41, 65, 91, 53]
HIGHEST_POSSIBLE_SCORE = 100

# O(n + k) If we didn't treat highest_possible_score as a constant,
#  we could call it k and say we have O(n+k) time and O(n+k) space. 
def highScoresBucket(arr):

    res = [0] * 100
    res2 = [0] * len(arr)
    for i in arr:
        res[i] +=1
    print(res)
    
    j=0
    for i in range(len(res))[::-1]:

        if res[i] > 0:
            res2[j] = i
            j+=1
    return res2

print(highScoresBucket(unsorted_scores))
print("here here here")
'''
n = input()
a = raw_input().split()
'''
a = [37, 89, 41, 65, 91, 53]
val = [[i] * a.count(i) for i in range(100)]
print(val)
r = sum( val, [])
print(r)

a = [37, 89, 41, 65, 91, 53]
def highScoresBucket(arr):

    res = [[] for i in range(100)]
    
    for i in arr:
        res[i].append(i)
    print(res)
    
    j=0
    
    r = sum( res, [])
    return r

print(highScoresBucket(unsorted_scores))
