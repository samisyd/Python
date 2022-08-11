'''
You are given some numeric string as input. Convert the string you are
given to an integer. Do not make use of the built-in "int" function.
Example:
    "123" : 123
    "-12332" : -12332
    "554" : 554
    etc.

'''

def strtointfunc(strval):

    is_negative = False
    start_idx = 0
    if strval is None:
        return 0
    if strval[0] == "-":
        is_negative = True
        start_idx = 1

    val = 0
    for i in range(start_idx, len(strval)):
        
        mf = 10 ** (len(strval) - (i+1))
        digit = ord(strval[i]) - ord('0')
        val = val + (mf * digit)

    if (is_negative):
        val = -1 * val
    print(val)


#convert str to int
strtointfunc('233')

def inttostr(intval):

    # use arrays rather than actual strings because string creates a new copy everytime we append to it.
    # but arrays are dynamically allocated internally so less of overhead to create new copies 
    strval = []
    while (intval ):

        modval = intval % 10
        intval = intval // 10

        strval.append(chr(ord('0') + modval)) 
    #strval = strval[::-1]
    rstr = "".join(strval[::-1])
    print(rstr)

inttostr(234)

'''
An array is "cyclically sorted" if it is possible to cyclically shift
its entries so that it becomes sorted.
The following list is an example of a cyclically sorted array:

    A = [4, 5, 6, 7, 1, 2, 3]

Write a function that determines the index of the smallest element
of the cyclically sorted array.
'''

def sortArray(inputArr):

    low = 0
    high = len(inputArr) - 1
    #mid = (high + low) // 2  

    while low < high:

        mid = (high + low) // 2
     
        
        if inputArr[high] < inputArr[mid]:
            low = mid + 1
           
        elif inputArr[high] >= inputArr[mid]:
            high = mid 
     

   
    return low


val = 0
A = [4, 5, 6, 7, 1, 2, 3]
A2 = [5, 6, 7, 1, 2, 3, 4]
A3 = [6, 7, 1, 2, 3, 4, 5]
val = sortArray(A)
#print(A[val])



'''
write a function that takes a non-negative integer and returns the largest integer whose square is less than or equal to
the integer given:

Example:

Assume input is integer 300.
    
Then the expected output of the function should be 17 since 17 squared is 289 which is strictly less than 300. 
Note that 18 squared is 324 which is strictly greater than 300, so the number 17 is the correct response.
'''

def getRoot(val):

    low = 0
    high = val
    lowestmid = 0

    while low < high:
        
        mid = (high + low )// 2
        if mid** 2 > val :
            high = mid            
        elif mid** 2 < val:
            low = mid
            if lowestmid == low:
                return lowestmid
            lowestmid = low
            
        elif mid == val:
            return mid

    return low

retVal = getRoot(300)
print(retVal)

'''
Define a bitonic sequence as a sequence of integers such that:
    x_1 < ... < x_k > ... > x_n-1 for some k, 0 <= k < n.
For example:
    1, 2, 3, 4, 5, 4, 3, 2
is a bitonic sequence. Write a program to find the largest element in such a
sequence. In the example above, the program should return "5".
We assume that such a "peak" element exists.
"""

# Peak element is "5".
A = [1, 2, 3, 4, 5, 4, 3, 2, 1]
'''

A = [1, 2, 3, 4, 5, 4, 3, 2, 1]
#A = [1,6,5,4,3,2,1]
def bitonicPeaak(arr):

    low = 0
    high = len(A) - 1

    while (low < high):

        mid = (low + high) // 2

        midlow = mid - 1
        midhigh = mid  +1

        if A[mid] < A[midhigh] and A[mid] > A[midlow]:
            low = midhigh
        elif A[mid] > A[midhigh] and A[mid] < A[midlow]:
            high = midlow
        elif A[mid] > A[midhigh] and A[mid] > A[midlow]:
            return mid

val = bitonicPeaak(A)    
print(val, A[val])


"""
Given two strings, write a method to decide if 
one is a permutation of the other.
"""

is_permutation_1 = "google"
is_permutation_2 = "ooggle"

not_permutation_1 = "not"
not_permutation_2 = "top"

def is_perm2(str1, str2):

    str1 = "".join(sorted(str1))
    str2 = "".join(sorted(str2))

    print(str1, str2)

    if len(str1) != len(str2):
        return False
    
    n = len(str1)
    for i in range(n):
        if str1[i] != str2[i]:
            return False
    return True

def is_perm(str1, str2):

    str1 = str1.lower()
    str2 = str2.lower()

    dict1 = dict()

    for i in str1:
        dict1[i] = 1 + dict1.get(i, 0)

    print(dict1)
    for i in str2:
        if i in dict1:
            dict1[i] -=1
        
    
    print(dict1)

    for val in dict1.values():
        if val != 0:
            return False
    return True

#print(is_perm2(is_permutation_1, is_permutation_2))
#print(is_perm2(not_permutation_1, not_permutation_2))

"""
Given a string, write a function to check if it is
a permutation of a palindrome. A palindrome is a word
or phrase that is the same forwards and backwards.
A permutation is a rearrangement of letters. The
palindrome does not need to be limited to just
dictionary words.
"""

palin_perm = "Tact Coa"
not_palin_perm = "This is not a palindrome permutation"

def is_palin(str1):

    dict1 = dict()

    str1 = str1.lower()
    str1 = str1.replace(" ", "")
    n = len(str1)
    for i in str1:
        dict1[i] = 1+dict1.get(i, 0)
        
    print(dict1)

    count = 0
    for k,v in dict1.items():
        
        if v % 2 != 0 :
            if count == 0:
                count = 1
            else:
                return False

    return True

#print(is_palin(palin_perm))
#print(is_palin(not_palin_perm))

# calculate string length

def str_len(str1):

    count = 0
    for i in str1:
        count +=1

    print(count)

def str_len_rec(str1):

    if len(str1) == 0:
        return 0
    else:
        return 1 + str_len_rec(str1[1:])



#print(str_len_rec("samina"))

vowels = "aeiou"

input_str = "abc de"
#input_str = "LuCiDPrograMMiNG"

def str_conso(str1):
    
    cons_ct = 0
    n = len(str1)
    for i in range(n):
        if str1[i].lower() not in vowels and str1[i].isalpha():
            cons_ct += 1

    print(cons_ct)


str_conso(input_str)

#look and say sequence

def lookandsay(str1):

    n = len(str1)
    #for i in range(n):
    i = 0
    arr1 = []
    while i < n:

        count = 1 
        while i+1 < n and str1[i] == str1[i+1]:
            count +=1
            i+=1

        arr1.append(str(count)+str1[i])        
        i+=1
    
    arr1 = "".join(arr1)
    print(arr1)


lookandsay("1211")

A = [2,4,3,1,2,3,9,9,1]
def findOddNum(nums):

    result = 0
    for i in nums:
        result ^= i

    return result

print(findOddNum(A))

Arr = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
"""
for row in Arr:
    for item in row:
        print(item)
"""
print("printings array with cols and rows")
for i in range(len(Arr)):
    for j in range(len(Arr[i])):
       print(Arr[j][i])


#Arr[1][2] = 99
#print(Arr)

result = 0
for i in range(len(Arr)):
    for j in range(len(Arr[i])):
        if i == j:
            result += Arr[i][j]
print(result)



#A = [1,2,3,4,5]
#print(A[2:])
#print(A[:2])

# given 2 arrays , find their intersection


arr1 = [4,9,5]
arr2 = [9,4,9,8,4]

# sets ==> union / | - Represents all elements in both sets
# differnce / - Represents elements in first set that is not present in second set
# x1.symmetric_difference(x2) and x1 ^ x2 return the set of all elements in either x1 or x2, but not both:
# x1.isdisjoint(x2) Determines whether or not two sets have any elements in common. - returns true if nothing common
# x1.issubset(x2) x1 <= x2  Determine whether one set is a subset of the other.

def intArrIntersect(arr1,arr2):

    #arr1 = set(arr1)
    # both work intersection or &

    #val = set(arr1).intersection(arr2) 
    val = set(arr1) & set(arr2)
    print("intersect val =", list(val))


intArrIntersect(arr1,arr2)

def intArr(arr1,arr2):

    result = []
    arr1.sort()
    arr2.sort()
    
    i =0 
    j = 0
    while i < len(arr1) and j < len(arr2):

        if arr1[i] == arr2[j]:
            result.append(arr1[i])
            i+=1
            j+=1
        elif arr1[i] < arr2[j]:
            i+=1
        elif arr1[i] > arr2[j]:
            j+=1
    return result

num1 = [1,2,2,1]
num2 = [2,2]
print(intArr(arr1,arr2))

#transpose of a matrix

matrix = [
    [3, 5, 8],
    [7, 1, 9],
    [2, 0, 4]
]

def rotateMat(Arr):

    rows = len(Arr)
    cols = len(Arr[0])
    for i in range(rows):
        for j in range(i, cols):
            Arr[i][j], Arr[j][i] = Arr[j][i], Arr[i][j]

    print(Arr)

    for i in range(rows):
        Arr[i][rows-1], Arr[i][0] = Arr[i][0],Arr[i][rows-1] 

    print(Arr)

print("rotating matrix")
rotateMat(matrix)

#hour glass problem
#https://www.hackerrank.com/challenges/2d-array/problem
mat2 = [
    [1, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 2, 4, 4, 0],
    [0, 0, 0, 2, 0, 0],
    [0, 0, 1, 2, 4, 0]
]

'''
for i in range(6):
    row = input().strip().split(' ')
    row = list(map(int, row))
    grid.append(row)
'''
arrlen = len(mat2)
result = -63
for i in range(1, 5):
    for j in range(1,5):

        sum = 0
        sum = sum + mat2[i-1][j-1]
        sum = sum + mat2[i-1][j]
        sum = sum + mat2[i-1][j+1]
        sum = sum + mat2[i][j]
        sum = sum + mat2[i+1][j-1]
        sum = sum + mat2[i+1][j]
        sum = sum + mat2[i+1][j+1]

        result = max(result, sum)

print(result)


# Complete the hourglassSum function below.
def hourglassSum(arr):
    
    sum = []
    
    for i in range(len(arr)-2):
        for j in range(len(arr)-2):
            
            sum.append(arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + 
                        arr[i+2][j] + arr[i+2][j+1] + arr[i+2][j+2])
    return max(sum)

'''
https://www.hackerrank.com/challenges/array-left-rotation/submissions/code/206766680
left rotation operation on an array of size shifts each of the array's elements 
unit to the left. Given an integer, , rotate the array that many steps left and return the result. 

d = 2
arr = [1,2,3,4,5]
after 2 rotations => arr = [3,4,5,1,2]
'''

def rotateLeft(d, arr):
    # Write your code here
    res = arr[d:] + arr[:d]
    print(res)
    return res




#dfs

Graph = {
     1: [0 , 3],
     0: [1, 3, 2],
     2: [0, 3],
     3: [1, 0, 4],
     4: [3]
}

Graph2 = {
     1: [5 , 7],
     4: [8],
     5: [1, 8, 12],
     7: [1, 16],
     8: [5, 12, 14, 4],
     12: [5, 8, 14],
     14: [12, 8],
     16: [7]
}

visited = set()
def dfsGraphs(Graph, start, visited):

    if visited is None:
        visited = set()
    if start in visited:
        return
    visited.add(start)
    print(start)
    for neigh in Graph[start]:
        dfsGraphs(Graph, neigh, visited)
    

def bfsGraphs(Graph, start, visited):

    queue = [start]
    i = 0
    visited.add(start)
    while i < len(queue):

        popped = queue[i]
        print(popped)
        i+=1
        for neigh in Graph[popped]:
            if neigh not in visited:
                queue.append(neigh)
                visited.add(neigh)
        


print("printing BFS")
#bfsGraphs(Graph2, 5, visited)
print("")

print("printing DFS")
#dfsGraphs(Graph2, 5, visited)
print("")

#for k,v in Graph.items():
 #   print(k,v)


#bfs
import collections

Grp = {
    'a': ['c', 'b'],
    'b': ['d'],
    'd': ['f'],
    'c': ['e'],
    'e': [],
    'f': []
}
    
output = []
visited = set()
def dfs(Graph, root):   
    
    visited.add(root)
    output.append(root)
    for vertex in Graph[root]:
        if vertex not in visited:
            dfs(Graph, vertex)
    
    return output


def dfs2(Graph, root):
   visited = set()
   #visited.add(root)
   stack = [root]
   result = []
   while len(stack):
       v = stack.pop()
       visited.add(v)
       result.append(v)
       for vertex in Graph[v]:
           if vertex not in visited:
               stack.append(vertex)
   return result


print("printing dfs", dfs(Grp, 'a'))
print("printing dfs2", dfs2(Grp, 'a'))

def bfs(Graph, root):

    visited = set()
    queue = collections.deque([root])
    result = []
    while queue:

        vertex = queue.popleft()
        visited.add(vertex)
        result.append(vertex)
        for neighb in Graph[vertex]:
            if neighb not in visited:
                queue.append(neighb)
                print(neighb , queue)
    print(result)

bfs(Grp, 'a')



#sliding window problem  - Maximum Size SubArray Of Size K

Arr = [1, 9, -1, -2, 7, 3, -1, 2]

def slidw(Arr, k):

    max_size = 0
    for i in range(k):
        max_size += Arr[i]

    for i in range(1, len(Arr) - k):

        sum = max_size - Arr[i-1] + Arr[i+k-1]
        max_size = max(sum, max_size)
    
    print(max_size)

print("printing sliding window",slidw(Arr, 3))

print("Geeks : %10d, Portal : %2.6f" %(1, 05.333) )

print(1 ^2 ^ 1)


'''
A queue is an abstract data type that maintains the order in which elements were added to it,
allowing the oldest elements to be removed from the front and new elements to be added
to the rear. This is called a First-In-First-Out (FIFO) data structure because the first
element added to the queue (i.e., the one that has been waiting the longest) is 
always the first one to be removed.

A basic queue has the following operations:

    Enqueue: add a new element to the end of the queue.
    Dequeue: remove the element from the front of the queue and return it.

Sample Input

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

42, 14,28,60,78
Sample Output

14
'''

def QueueImpementation():
    noele = input()
    queue = []
    str1 = ""
    for i in range(int(noele)):    
        str1 = input()
        res = list(map(int, str1.split()))
        print(res)
        
        if res[0] == 1:
            print("adding", res[1] )
            queue.append(res[1])
        elif res[0] == 2:
            print("removing", queue[0] )
            if len(queue):
                queue.pop(0)
        elif res[0] == 3:
            if len(queue):
                print("print top of queue",queue[0])
        else:
            print("invalid input")

#print("starting queue imp", QueueImpementation())

#
# Complete the 'isBalanced' function below.
#
# The function is expected to return a STRING.
# The function accepts STRING s as parameter.
#
#  first s = '{[()]}'
#  second s = '{[(])}' 
#  third s ='{{[[(())]]}}'
def isBalanced(s):
    # Write your code here
    stack = []
    setOpen = {'{','(', '['}
    setClose = {')', '}', ']'}
    
    for item in s:
        if item in setOpen:
            stack.append(item)
        elif item in setClose and not len(stack) :
            return 'NO'
        elif item == ']' :
            if '[' != stack.pop():
                return 'NO'
        elif item == '}' :
            if '{' != stack.pop():
                return 'NO'
        elif item == ')' :
            if '(' != stack.pop():
                return 'NO'
    
    
    if not len(stack):
        return 'YES'
    else:
        return 'NO' 

stack = "samina345678"
val = stack[-6:]
print(val)
stack = stack[:-6]
print(stack)

'''
https://www.hackerrank.com/challenges/one-week-preparation-kit-simple-text-editor/problem
Implement a simple text editor. The editor initially contains an empty string, . Perform operations of the following types:
append - Append string to the end of
delete - Delete the last characters of
print - Print the character of
undo - Undo the last (not previously undone) operation of type or , reverting to the state it was in prior to that operation. 

operation
index   S       ops[index]  explanation
-----   ------  ----------  -----------
0       abcde   1 fg        append fg
1       abcdefg 3 6         print the 6th letter - f
2       abcdefg 2 5         delete the last 5 letters
3       ab      4           undo the last operation, index 2
4       abcdefg 3 7         print the 7th characgter - g
5       abcdefg 4           undo the last operation, index 0
6       abcde   3 4         print the 4th character - d
'''

def StrEdi():
    noOps = input()

    strVal = ""
    strIn = ""
    stack = []

    for i in range(int(noOps)):
        
        strIn = input()
        res = strIn.split()
        command = int(res[0])
        
        if command == 1:
            #append
            stack.append(strVal)
            strVal+=res[1]        
            
        elif command == 2:        
            #delete last k chars
            stack.append(strVal)
            k = int(res[1])
            strVal = strVal[:-k]        
            
        elif command == 3:    #print the chracter asked
            k = int(res[1])-1        
            print(strVal[k])
        elif command == 4:
            strVal = stack.pop()             
        else:
            print("invalid input")
    

def StrEdi():
        
    noOps = input()

    noOps = int(noOps)
    strVal = ""
    stringEd = ""
    lastOp = []
    for i in range(noOps):
        
        strVal = input()
        res = strVal.split()
        command = int(res[0])
        
        if command == 1: #append        
            stringEd += res[1]
            lastOp.append(res)
            
        elif command == 2: #delete last k chars
            k = int(res[1])
            deletedStr = stringEd[-k:]
            stringEd = stringEd[:-k]
            data = [str(command), deletedStr]
            lastOp.append(data)
                        
        elif command == 3:        # print the character asked
            k = int(res[1])-1
            if len(stringEd) > k:
                print(stringEd[k])
        elif command == 4: # undo last operation
            result = lastOp.pop()
            undoC, val = result[0], result[1]
            undoC = int(undoC)
            if undoC == 1: #undo append
                lstr = len(val)
                if len(stringEd) >= lstr:
                    stringEd = stringEd[:-lstr]
            elif undoC == 2: #undo delete
                stringEd += val        
        else:
            print("invalid input")
        
        
'''
https://www.hackerrank.com/challenges/dynamic-array/problem

Function Description
Complete the dynamicArray function below.
dynamicArray has the following parameters:
- int n: the number of empty arrays to initialize in
- string queries[q]: query strings that contain 3 space-separated integers

Returns -int[]: the results of each type 2 query in the order they are presented

'''
def dynamicArray(n, queries):
    # Write your code here
    
    lastAnswer = 0
    arr1 = {}    
    queryType = 0    
    lis3 = []
    lis4 = []
    lis = []
    result = []
    
    #for j in range(n):
     #   lis3 = lis4.copy()
      #  lis.append(lis3)
    lis = [[] for j in range(n) ]
        
    for i in range(len(queries)):
        
        queryType = queries[i][0]
        val = (queries[i][1] ^ lastAnswer ) % n
        if (queryType == 1):            
            lis[val].append(queries[i][2])            
            arr1[val] = lis[val]            
        elif (queryType == 2):            
            y = queries[i][2] % len(arr1[val])
            lastAnswer = arr1[val][y]
            result.append(lastAnswer)            
        else:
            return 0
    print('result', result)    
    return result