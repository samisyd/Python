import collections
import queue
from re import sub

'''
#https://www.youtube.com/watch?v=xPBLEj41rFU

DP 29. Minimum Insertions to Make String Palindrome
1. get the  - longest palindromic subsequence
2. answer is ==> n - longest palindromic subsequence 

ex - codingninjas

How do u approach ?
==> keep the palindromic position intact

cod ingni njas
ingni - palindromic subsequence

cod sajn ingni njas doc - 12 - 5 = 7 insertions needed

Longest Palindromic Substring - Python - Leetcode 5
https://www.youtube.com/watch?v=XYQecbcd6_c

LeetCode 2: Minimum swaps to make palindrome
https://www.youtube.com/watch?v=fOpD3wdK0w8

Palindromic Substrings - Leetcode 647 - Python
https://www.youtube.com/watch?v=4RACzI5-du8

Given a string s, return the no of palindromic substrings in it.
input = 'abc'
output = 3
the 3 palindromic substrings are 'a', 'b', 'c'

ex:2
input s = 'aaa'
output = 6
values are - a , a ,a ,aa, aa, aaa
'''

def getPalindromStr(s):

    m = len(s)
    res = 0

    # Use greeedy method to get the odd and even palindrome substrings
    for i in range(len(s)):
        
        # checking for odd number of strings
        # checking 1 string and add to result
        # then strtch outward and check for match
        l = r = i
        while l >=0 and r < len(s) and s[l] == s[r]:
            res +=1
            l -=1
            r += 1

        # checking for even number of strings
        # check initial 2 strings and stretch outwards from both sides if within boundries and until equal and count
        # then check the next 2 strings
        l = i 
        r = i+1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            res += 1
            l -=1
            r +=1

    return res


print("getPalindromStr for abc")
print(getPalindromStr("abc"))

print("getPalindromStr for aaa")
print(getPalindromStr("aaa"))

'''
https://www.youtube.com/watch?v=XYQecbcd6_c
Longest Palindromic Substring - Python - Leetcode 5
input - babad
output - bab, or aba

input - cbbd
o/p - bb
'''
def longestCommSubstr(s):

    res = ""

    for i in range(len(s)):

        # find odd length palindromic substrings 
        l = r = i
        while l>=0 and r<len(s) and s[l] == s[r]:
            if len(res) < r-l+1:
                res = s[l:r+1]
            l -=1
            r +=1 
        
        #find even length palindromic substrings
        l = i
        r = i+1
        while l>=0 and r<len(s) and s[l] == s[r]:
            if len(res) < r-l+1:
                res = s[l:r+1]
            l -=1
            r +=1

    return res 

print("longestCommSubstr")
print(longestCommSubstr('cbbd'))
print(longestCommSubstr('babad'))
print(longestCommSubstr('bad'))

def getCons(index, arr, visited):

    rows = len(arr)
    cols = len(arr[0])

    row,col = index
    sum = 0

    if arr[row][col] == 0:
        return 0

    arrList = collections.deque([(row,col)])
    sum = arr[row][col]
    visited.add((row,col))
    directions = [[-1,0],[1,0],[0,-1],[0,1]]
    while len(arrList):
        
        row, col = arrList.popleft()
        for dr,dc in directions:
            
            nrow, ncol = row+dr, col+dc            
            if ((nrow,ncol) not in visited) and nrow >= 0 and nrow < rows and ncol >=0 and ncol < cols and arr[nrow][ncol] != 0:
                sum += arr[nrow][ncol]
                arrList.append([nrow, ncol]) 
                print("print arrlist and sum", arrList, sum)        

    return sum


arr = [ 
        [0, 2, 3, 4, 1],
        [0, 4, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 2, 3, 0, 0],
        [0, 2, 0, 0, 0]
    ]

'''
keep all the indexes in the main set which are non zero
while values are there in main set
pop one from the set and do a bFS of the index and get the non-zero neighbors 
-- use a queue to do a bfs and 
-- store only the neighbors in queue and visited set which are not already in visited set - avoid dups
take difference of main set with the visited set
again continue the bfs with the left set until it gets empty 
'''

def getMaxCons(arr):
    
    rows = len(arr)
    cols = len(arr[0])
    results = []

    # create a set that has the row,col pair which is not 0
    filledindexarr = set()
    for i in range(rows):
        for j in range(cols):
            if arr[i][j] != 0:
                filledindexarr.add((i,j))
    print(filledindexarr)

    while len(filledindexarr) :
        index = filledindexarr.pop()
        visited = set()
        sum = 0
        sum = getCons(index, arr, visited)
        results.append(sum)
        filledindexarr = filledindexarr.difference(visited)
    print(results)
    return max(results)

#print("max const is ",getMaxCons(arr))


def binarysearch(arr, n, k):
		# code here
		
    low = 0
    high = n-1
    
    while(low <= high):
        
        mid = (low+high)//2
        
        if arr[mid] == k:
            return mid
        elif arr[mid] < k:
            low = mid+1
        else:
            high = mid-1
            
    return -1
arr = [1,2,3,4,5]
k = 4
print(binarysearch(arr, 5, k))


#Function to reverse every sub-array group of size k.
def reverseInGroups(arr, N, K):
    # code here
    
    i = 0 
    # if K is bigger than N then make j to be N-1
    j = K-1 if K <= N else N-1

    start = 0
    while start < N: 
        while i<j:
            arr[i], arr[j] = arr[j], arr[i]
            i +=1
            j -=1
        
        print(arr)
        start += K
        i = start
        j = start + K-1 
        if j >= N:
            j = N-1


arr = [1,2,3,4,5]
reverseInGroups(arr, 5, 3)
print(arr)


#Function to check if two arrays are equal or not.
def check(A,B,N):
    
    #return: True or False
    C=dict()
    #code here
    for item in A:
        C[item] = 1+ C.get(item, 0)
    
    #print(C)
    for item in B:
        if item in C:
            C[item] -=1
        else:
            return 0
    
    #print(C)
    for value in C.values():
        if value >= 1:
           # print("found 1")
            return 0
    
    return 1

A = [3,3]
B = [2,2]
print(check(A,B,2))

arr = [3, -7, 0]
def minimumAbsoluteDifference(arr):

    
    mindiff = float('inf')
    # Write your code here
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            diff = abs(arr[i] - arr[j])            
            mindiff = min(diff, mindiff)            
    
    return mindiff

print(minimumAbsoluteDifference(arr))
#Complexity - O(n)
# first sort the array. the closest elements will be the min diff. 
# so just take diff between each of the 2 neighbor elements. 
def minimumAbsoluteDifference(arr):
    arr.sort()
    mindiff = float('inf')
    # Write your code here
    for i in range(1, len(arr)):
        diff = abs(arr[i] - arr[i-1])
        mindiff = min(diff, mindiff)

    return mindiff

def marcsCakewalk(calorie):
    # Write your code here
    calorie.sort()
    sum = 0
    exponent = 0
    # reverse the array 
    for item in calorie[::-1]:
        sum += item * 2**exponent
        exponent += 1
    
    return sum

'''
Given a square grid of characters in the range ascii[a-z], rearrange elements of each 
row alphabetically, ascending. Determine if the columns are also in ascending alphabetical 
order, top to bottom. Return YES if they are or NO if they are not.
'''
grid = ['abc','ade', 'efg']
grid2 = ['abc',
        'hjk', 
        'mpq',
        'rtv']

def gridChallenge(grid):
    # Write your code here
    rows = len(grid)
    cols = len(grid[0])

    for i in range(rows):
        grid[i] = sorted(grid[i])
    
    
    for i in range(1, rows):
        for j in range(0, cols):
            if grid[i-1][j] > grid[i][j]:
                return False

    return True

print(gridChallenge(grid2))


'''
You have three stacks of cylinders where each cylinder has the same diameter,
 but they may vary in height. You can change the height of a stack by removing
  and discarding its topmost cylinder any number of times.

Find the maximum possible height of the stacks such that all of the stacks 
are exactly the same height. This means you must remove zero or more cylinders
from the top of zero or more of the three stacks until they are all the same 
height, then return the height. 
https://www.hackerrank.com/challenges/equal-stacks/problem
'''
def equalStacks(h1, h2, h3):
    # Write your code here
    
    s1,s2,s3 = map(sum, (h1,h2,h3))
    
    h1,h2,h3 = h1[::-1], h2[::-1], h3[::-1]
    
    while h1 and h2 and h3:
        
        minSize = min(s1,s2,s3)
        if s1 > minSize : s1 -= h1.pop() 
        if s2 > minSize : s2 -= h2.pop()
        if s3 > minSize : s3 -= h3.pop()
        if s1 == s2 == s3: return s1
    return 0



'''
Next greater element in an array 

1. Brute force T - O(n^2) - iterate through each n times and get the greaater ele
2. Use new stack T - O(n) - push the indexes of the element of arr
Rule - Smaller or equal elements to be pushed on top of stack
'''
#https://www.youtube.com/watch?v=sDKpIO2HGq0&list=PLEJXowNB4kPzEvxN8ed6T13Meet7HP3h0
#https://www.youtube.com/watch?v=68a1Dc_qVq4

arr = [13, 7, 6, 12, 10]
arr2 = [4, 1, 4, 2]
def GretEle(arr):

    res = [-1 for i in range(len(arr))]

    n = len(arr)
    for i in range(n):
        for j in range(i+1,n):
            if arr[i] < arr[j]:
                res[i] = arr[j]
                break
    
    print(res)

#using stack
#2. Use new stack T - O(n) - push the indexes of the element of arr
#Rule - Smaller or equal elements to be pushed on top of stack
def GretEle2(arr):

    n = len(arr)
    #initialize results
    res = [-1 for i in range(n)]
    # stack will contain the index of element in the array
    stack = [0]

    for i in range(1, n):

        #if next arr ele is smaller to the top of stack element then push in stack
        if arr[i] <= arr[stack[-1]] : 
            stack.append(i)            
        else:
            # while stack top element is smaller to arr ele then pop from stack and fill the result arr
            while stack and arr[stack[-1]] < arr[i]: 
                index = stack.pop()
                res[index] = arr[i]
            stack.append(i)
        
    print(res)

print("inside optimized greater element  ")
GretEle(arr)
GretEle2(arr2)

'''
https://www.hackerrank.com/challenges/game-of-two-stacks/forum

Alexa has two stacks of non-negative integers, stack and stack where index 0
denotes the top of the stack. Alexa challenges Nick to play the following game:
    In each move, Nick can remove one integer from the top of either stack a[n]
or stack b[m]
.
Nick keeps a running sum of the integers he removes from the two stacks.
Nick is disqualified from the game if, at any point, his running sum becomes greater than some integer
given at the beginning of the game.
Nick's final score is the total number of integers he has removed from the two stacks.
Consider two stacks s1 = [17,1,1,1,8] and s2 = [8,8,4,5,9] and max sum = 20

17+1+1+1 = 20 - takes total 4 elements

2nd ex - max sum = 11
<4,2,4,6,1 > and <2,1,8,5> 

4+2+4 = 10

4+2+ (drop 4 of 1st stack) and add 2 from other stack
4+2+2 = 8 < 10
add 1 more frm 2nd stack
4+2+2+1 = 9

4+2+2 (drop 2 of 1st stack) and add 1 from other stack
4+2+2 = 8

on hcckerrank it doesnt passs all test cases
Solution:


'''
a1 = [4,2,4,6,1]
a2 = [2,1,8,5]
def gameof2stack(maxval, a,b):
    # Keep Adding elements of 1st stack until it is less than maxval 
    # and count the no of ele added. store in result  
    a = a[::-1]
    b = b[::-1]
    atmp = []
    totalsum = 0
    for i in range(len(a)):
        if totalsum <= maxval:
            val = a.pop()
            totalsum += val            
            atmp.append(val)
        else:
            break
    res = len(atmp)
    

    # now take ele from stack B and add total and keep removing ele of stack A until it is <= total
    # return the max ele found from both the methods
    currCount = res
    m = len(b)
    while m and len(atmp):
    
        if totalsum + b[-1] < maxval:
            totalsum += b.pop()
            currCount +=1
            m -=1
            if currCount  > res:
                res = currCount 
            continue
        
        totalsum -= atmp.pop()
        currCount -=1

    return res 

print(gameof2stack(10, a1,a2))

def twoStacks(maxSum, a, b):
    # Write your code here
    res = total = i = j = 0
    # pick ele from stack a till <= maxSum
    while i < len(a) and total + a[i] <=  maxSum:
        total += a[i]
        i +=1
        res += 1
    
    while j < len(b) and i >= 0:
        
        total += b[j]
        j += 1
        while total > maxSum and i > 0:
            total = total - a[i]
            i -=1
        
        if total <= maxSum and res < i+j:
            res = i+j
    
    return res

'''
Rotten oranges
Goal : Find in how much time can all oraanges rot
2 - Rotten, 1 - Fresh, 0 - empty space
'''
arr = [ 
        [2, 1, 0, 2],
        [1, 0, 1, 2],
        [1, 0, 0, 1]
    ]

def rottenOr(arr):

    rows = len(arr)
    cols = len(arr[0])

    retTime = 0
    fresh = 0
    q = collections.deque()
    for i in range(rows):
        for j in range(cols):
            if arr[i][j] == 2:
                q.append([i,j])
            elif arr[i][j] == 1:
                fresh +=1
    
    print(q)
    visited = set()
    directions = [[1,0], [-1,0], [0,1], [0,-1]]
    while q and fresh > 0:

        qlen = len(q)
        while qlen:

            row, col = q.popleft()
            visited.add((row,col))
            for r,c in directions:
                nrow, ncol = row+r, col+c
                if (nrow, ncol) not in visited:
                    if nrow >= 0 and nrow < rows and ncol >=0 and ncol < cols:
                        if arr[nrow][ncol] == 1:
                            arr[nrow][ncol] = 2
                            fresh -=1
                            q.append([nrow,ncol])

            qlen -=1
            
        
        print("fresh is ",fresh)                    
        retTime +=1
        print(retTime)
        

    return retTime if fresh == 0 else -1


print(rottenOr(arr))

'''
Candy distributuion problem
https://www.youtube.com/watch?v=h6_lIwZYHQw

constraints:
1. n children
2. each one gets 1 candy
3. children with higher ratings gets more candies than neighbors

dividde the problem in 2 parts

children        12  4  3  11  34  34  1  67
each gets 1     1   1  1  1   1    1   1  1
LeftToRight     1   1  1  2   3   1   1  2
RightToLeft      3   2  1  1   1   2   1  1

total-maxofBoth  3  2  1  2  3  2  1 2  = 16 candies to give

'''

# starting cordinates are given and one needs to reaach the destination cordinates

import collections
def minimumMoves(grid, startX, startY, goalX, goalY):
    # Write your code here
    lenGrid = len(grid)
    print(lenGrid)
    cnt = 0
    #keep a visited set
    visited = set() 
    #Use a queue to do a BFS and get the neighboring cordinates to 
    # reach the destination  
    mqueue = collections.deque([[startX, startY, cnt]])
    
    directions = [[1,0],[-1,0],[0,1],[0,-1]]
    
    visited.add((startX, startY))
    while mqueue:
        pathx, pathy, c = mqueue.popleft()
        #print("visited", visited)
        
        c+=1
        for xi, yi in directions:
            x , y = pathx, pathy
            
            while True:
                x, y = x+xi, y+yi
                #print(x,y)
                if 0 <= x <lenGrid and 0 <= y <lenGrid and grid[x][y] == '.':
                 
                    if x == goalX and y == goalY:                        
                        return c
                    elif (x,y) not in visited:
                        
                        # add the cordinates in the queue if not in visited set
                        visited.add((x,y))
                        mqueue.append([x,y,c])
                        
                else:
                    break

grid = [
    '.X.',
    '.X.',
    '...'
]
startX, startY = 0,0
goalX, goalY = 0,2
print("minimumMoves")
print(minimumMoves(grid, startX, startY, goalX, goalY))

'''
https://leetcode.com/problems/first-unique-character-in-a-string/
Given a string s, find the first non-repeating character in it 
and return its index. If it does not exist, return -1.

Input: s = "loveleetcode"
Output: 2
'''
s = "loveleetcode"
def nonRepeatChar(s):

    lens = len(s)
    hashm = dict()

    for i in range(lens):
        hashm[s[i]] = 1 + hashm.get(s[i], 0)

    for i in range(lens):
        if hashm[s[i]] == 1:
            return i
    return -1

s2 = 'aabb'
s3 = 'leetcode'
print("nonRepeatChar")
print(nonRepeatChar(s))

# Given a string s, find the length of the longest substring without repeating characters.
# leetcodde 3 
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def LongestSubstringSolution(s):
    # Type your solution here
    m = len(s)
    l = 0
    
    if m == 0:
        return 0
    resSubStr = s[0]
    
    for r in range(1, m):
        substr = s[l:r+1]
        i = l 
        j = r
        fvalid = True
        while i <j and fvalid == True:
            if s[i] !=s[j]:
                i+=1
                fvalid = True
            else:
                fvalid = False
                l=i+1
                break
        if fvalid and len(resSubStr) < r-l+1:
            resSubStr = substr            

    return len(resSubStr)       

s = 'nndNfdfdf'
s5 = 'dvdf'
s2 = "uJJvmjCtiEsIFLNmECZhQUluhrQLUrjNerJJUQBujGNDFhpoHpjxMEhiyPJWkstZLGvSnkHIgQzkHXlYzeDeTvPjRmgmxtfuPeAyJvjNYiiyvIQpeRoUbOTSFrmIyofsPnVtkmwxCYxmaoBpcpxYpbQocLrijzgdGOrdQNjikkLmDonAUFtajWwmXhuAFmAHEEUBIXgDRSSzxFToDmrhCQjBMtSzrOjHfaQCyTslRZOwzbDLwuCEHVzGE"
s3 = "DcgaBpWTUzDnOHwbNpndVBLjjEnFUdkFsZyscPjnePIDJFCOjigGZgmosnjFRgqHyHjTookcuyZiCTSwgrarvVNwYNSFwcWUVGAKPFSgvEFYsfRwrJQGCJATvIdasAVAqmXzJkrPmeGJmbeoHmtHXFhUqsnEhJDZyrdJQgkBzuCsyJtCNDnHEYPsrUCZOUnbADZYIwuAPfxbssnYNXgibpcnSHUfbybjCqnsMGafGigbBkCHoVjUmqolz"
s4 = 'vMVynSPKemexiKjNERCtbDzoUmJaVtaVrqXVgEQftUEOWEFpwwkKTFfZZRIquuKwgmumuWNXUfpfvVdyfSBgZjIcugkqRUApDmGNmixaxkluDTPzKeMYOWMQizjtbfJGzZBGLWNyZrKQKAzuIMfCtnXtdgMTosCXqXNulWJZgghvHxnEfKoIDtgauFWCMYzmYkEepvKmqcUWniUxDRCJocKDLiYpJcDYKYEiRDoKtoyLMBWSoxnaKgFaNNWtBZOrtFoVuIfJBDpJHxKQBHFssoBkAdKfPEEXQIxhdWlxCzZOuGzCGPAmpRglOGpimJglZWDlBbPgAvtteKqjjEXZXeZdTVKPQgNpGPtHNSeLZBBrIobVNCOEimZHWNjEBdVZyBMfCGRytdAvIyiuRVFpuZjMwumJKZmzsZXAHZTWlWVYrLWUcKMIBdHbTiItxvnmEAhKvhthnNmgIwqKpQpZmAEqBtIFxeIfqfAncgeqJPExPQNoWpyzVksbZQytDiDbYHQhRfBBmOlrHaLmvXLntYaaFLLDXxNyUVxvbzNNhMRidNpacKaIfLDXpUhsNPaMVpoaJtFXvePqpbWiKUoAbypFbCbedZAqHhEaNrEDrNQziwzmhNRUUFnmCNnfOPyYygbpTYvbrlRXIIHQhxfEpzJTEhWXsfRvkQdoxDVGlRtNKvazSTRFCLvxeHvQDMZarSegyNPaTaqFKRwulDRxspiGYUBhlckkfFxYPqBaAmPKkXtvjOyNBiCBkjzbVgKTcRGRZZoVKUXNtavsuaQQCcgnKbaJlnEPEHcdQxOlgiJxGGSTSkCLJCAcOInUZUxYDixXotEbrQiJqBFYxrqYRaDZWkawRPmxrueEfJgXjutRwAPetFAkMzEYLFSDKvAHYOfVXGAFurwNaTJJnQpxaplfCmwOpqmtzFXkjdcBgxjGWXxYcRHjQeKbCqUCNMiAdQKxMqIamEFcrshFGBXtZTFcMUtScDTVVVXLhLsd ' 
s6 = 'bbbbb'
print(LongestSubstringSolution(s4))
s = 'nndNfdfdf'
# Given a string s, find the length of the longest substring without repeating characters.
# leetcodde 3 
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def LongestSubstringSolution2(s):

    m = len(s)

    charset = set()
    l = 0
    res = 0

    for r in range(m):
        while s[r] in charset:
            charset.remove(s[l])
            l +=1
        charset.add(s[r])
        res = max(res, len(charset))

    return res


print("2nd optimized solution using sets")
print(LongestSubstringSolution2(s4))

#https://www.geeksforgeeks.org/print-the-lexicographically-smallest-dfs-of-the-graph-starting-from-1/

def lexdfs(g_nodes, g_from, g_to, r, v):
    # Write your code here
    
    graph = {i:[] for i in range(g_nodes)}
    
    for j in range(len(g_from)):
        graph[g_from[j]].append(g_to[j])
        graph[g_to[j]].append(g_from[j])
    
    def dfs(s,e):
        
        visited = set()
        stack = [s]
        total = 0
        while stack:
            val = stack.pop()
            if val == e:
                return total
            if val not in visited:
                visited.add(val)           
            
            verValarr = graph[val]
            print('verValarr',verValarr)
            total +=1

            
            while verValarr:            
                minV = min(verValarr)
                print(min)
                verValarr.remove(minV)
                print(stack)
                if minV not in visited:
                    stack.append(minV)
                    print('appending',stack)
                    break
                else:
                    continue
        
    res = []
    for start,end in zip(v,r):
        
        print(start,end)
        tot = 0
        tot = dfs(start,end)
        res.append(tot)
        
    return res
        
'''
https://leetcode.com/discuss/interview-question/1163699/Lex-DFS-of-Amazon
https://www.geeksforgeeks.org/print-the-lexicographically-smallest-dfs-of-the-graph-starting-from-1/
'''

def lexdfs2(g_nodes, g_from, g_to, r, v):
    # Write your code here
    
    graph = {i:[] for i in range(g_nodes)}
    
    for j in range(len(g_from)):
        graph[g_from[j]].append(g_to[j])
        graph[g_to[j]].append(g_from[j])
    
    for i in range(g_nodes):
        graph[i].sort()
        
    def dfs(s,e):
        
        visited = set()
        stack = [s]
        total = 0
        visited.add(s)
        while stack:
            val = stack.pop()
            if val == e:
                return total
            
            for nei in graph[val]:
                if nei not in visited:
                    stack.append(nei)
                    visited.add(nei)
            
    res = []
    for start,end in zip(v,r):
        
        print(start,end)
        tot = 0
        tot = dfs(start,end)
        res.append(tot)
        
    return res


# hackerrank server selection problem
'''
arr = [
    [3, 1, 1],
    [1, 2, 3],
    [5, 1, 3],
    [2, 5, 4]
]

choose the maximum col between 2 rows and return the min value in that row
max comparision between rows = col - 1 = 2

3 , 1, 1 and 1, 2, 3 ==> max is [3, 2, 3]
3,1,1 and 5,1,3 ==> max is [5,1,3]
'''


petrolpumps = [[1,5], [10,3], [3,4]]
def truckTour(petrolpumps):
    # Write your code here
    
    n = len(petrolpumps)
    resindex = -1
    petsum = 0
    totcount = 0
    for i in range(len(petrolpumps)):
        
        pet, dis = petrolpumps[i]
        isSuff = pet - dis
        petsum += isSuff
        print("1", resindex, petsum)
        if petsum < 0:
            print("1", resindex)
            resindex = -1
            petsum = 0
            continue
        else:
            if resindex < 0: 
                print("2", resindex)
                resindex = i
            totcount +=1
    
    if resindex == -1:
        return -1
    
    if totcount < n:
        print(n - totcount)
        for i in range(n-totcount):
            pet, dis = petrolpumps[i]
            isSuff = pet - dis
            petsum += isSuff
            if petsum < 0:
                resindex = -1
            else:
                totcount +=1
    print("petsum",petsum)
    if totcount == n:
        return resindex
    else:
        return -1

print(truckTour(petrolpumps))

# Reverse a linked list

class LinkedListNode(object):

    def __init__(self, value):
        self.value = value
        self.next  = None
    
class LinkedList:

    def __init__(self, data=None):
        if data :
            node = LinkedListNode(data)
            self.head = node
        self.head = None


    def reverse(head):

        if head == None:
            return None
        if head.next == None:
            return head
        curr = head
        prev = None
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        head = prev 

        return head


# Enter your code here. Read input from STDIN. Print output to STDOUT
def decodeHuff(root, s):
    #Enter Your Code Here
    m = len(s)
    
    res = ""
    node = root
    for i in range(m):
        
        
        if s[i] == '0':
            node = node.left
            if node.left == None and node.right == None:
                res += node.data
                node = root
                
        elif s[i] == '1':
            node = node.right
            if node.left == None and node.right == None:
                res += node.data
                node = root
        else:
            print('invalid input')
    
    print(res)
    return res

class Node:
    
    def __init__(self,data):
        self.right = None
        self.left = None
        self.data = data
    
class myTree:
    
    def __init__(self, data):
        self.root = Node(data)   
            

# find all paths till leaf
def findAllPathstillLeaf(treenode):

    path = []
    paths = []
    node = treenode
    def findPaths(node, path, paths):

        if not node:
            return None

        path.append(node.data)
        print(path)
        if node.left == None and node.right == None:
            paths.append(path.copy())
        findPaths(node.left, path, paths)
        findPaths(node.right, path, paths)
        print("popping", path)
        path.pop()

    
    findPaths(node, path, paths)
    return paths

def decode(codes, encoded):
    # Write your code here
    mytree = myTree('start')
    head = mytree.root
    node = head
    for items in codes:
        vals = items.split('\t')
        val = vals[0]
        item = vals[1]
        mlen = len(item)
        node = head
        for i in range(mlen):            
            if item[i] == '1':
                #if this is the last value
                if i == mlen - 1:
                    if node.left:
                        node.left.data = val
                    else:                                        
                        node.left = Node(val)
                    print('added val to left', val)
                else:  
                    if not node.left:                    
                        cval = item[0:i+1]
                        node.left = Node(str(cval))
                node = node.left                    
            elif item[i] == '0':
                #if this is the last value
                if i == mlen - 1:
                    if node.right:
                        node.right.data = val
                    else: 
                        node.right = Node(val)
                    print('added val to right', val)
                else:
                    if not node.right:
                        cval = item[0:i+1]
                        node.right = Node(str(cval))
                node = node.right

    #print(findAllPathstillLeaf(head))
    res = ''
    retres =[]
    node = head
    print(encoded)    
    for i in encoded:
        if i == '1':
            node = node.left
            if node and node.left == None and node.right == None:
                #found value                
                if 'newline' in node.data:                  
                    retres.append(res)
                    res = ''                                         
                else:
                    res += node.data
                node = head
                        
        elif i == '0':
            node = node.right
            if node and node.left == None and node.right == None:
                #found value                
                if 'newline' in node.data:                    
                    retres.append(res)
                    res = ''                    
                else:
                    res += node.data                
                node = head            
        else:
            print('invalid input')
        
    retres.append(res)    
    return retres
        

codes = ['a\t100100','b\t100101','c\t110001','d\t100000','[newline]\t111111','p\t111110','q\t000001'] 
encoded = '111110000001100100111111100101110001111110'
'''
111110 000001 100100 111111 100101 110001 111110
res = pqa, bcp
'''
print(decode(codes, encoded))

'''
palindrome index, determine the  index of the character
that can be removed to make the string a palindrome
s = aaab == > o/p 3
s = baa
s = aaa
0/p = 3, 0 , -1
'''

str = 'quyjjdcgsvvsgcdjjyq'
# o/p =  yjjdcgsvvsgcdjjy

# 2 pointer solution
def palindromeIndex3(s):

    m = len(s)
    l = 0
    r = m-1
    s1 = ''
    badindex1 =  badindex2= 0
    res = []

    def checkIsPal(s, res):
        l = 0
        r = len(s) -1
        while l<r:
            if s[l] == s[r]:
                l+=1
                r-=1
            else:
                res.append([l,r])
                return False
        return True

    val = checkIsPal(s, res)
    if val:
        return -1
    else:
        l,r = res[0][0], res[0][1]
        s1 = s[l+1:r+1]
        badindex1 = l
        badindex2 = r    
    
    val = checkIsPal(s1, res)
    if val:
       return badindex1 
    else:
        return badindex2 
    
print(palindromeIndex3('qyjjdcgsvvsgcdjjyuq'))
print(palindromeIndex3('aaa'))


print('todo')
arr = [1,2,3,4]
for i in range(len(arr)-1)[::-1]:
    print(i, end=' ')
