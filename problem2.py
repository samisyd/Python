'''
There is a given list of strings where each string contains only lowercase letters from a - j inclusive.
 The set of strings is said to be a GOOD SET if no string is a prefix of another string. In this case,
  print GOOD SET. Otherwise, print BAD SET on the first line followed by the string being checked.
Note If two strings are identical, they are prefixes of each other.
Example
Here 'abcd' is a prefix of 'abcde' and 'bcd' is a prefix of 'bcde'. Since 'abcde' is tested first, print
arr = ['aab','aac','aacghgh','aabghgh'] => o/p = 'BAD SET' aacghgh
arr = ['bcde', 'abcd', 'abcde', 'bcd'] ==> o/p = 'BAD SET' bcd 
'''

def noPrefix(words):
    
     # Write your code here
    
    foundbad = False
    lwords = len(words)
    res = []
    for i in range(lwords):
        for j in range(i+1, lwords):
            
            m = len(words[i])
            n = len(words[j])
            
            k = 0
            goodset = False
            while k < n and k < m:
                if words[i][k] == words[j][k]:
                    k+=1
                else:
                    goodset = True                    
                    break 
            
            if goodset == False:
                foundbad = True
                if k < m:
                    res.append([i, words[i]])
                else: 
                    res.append([j, words[j]])                
                break
                    
        
    if foundbad == False or lwords == 1:
        print('GOOD SET')
    else:
        print('BAD SET')
        res.sort()
        print(res[0][1])


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'noPrefix' function below.
#
# The function accepts STRING_ARRAY words as parameter.
#

class TrieNode:
    
    def __init__(self):
        self.children = {}
        self.endofWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        curr = self.root
        alpresent = False    
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = TrieNode()
            curr = curr.children[ch]
            if curr.endofWord == True:
                alpresent =True
        curr.endofWord = True
        
        if alpresent:
            return False
        return True
                
    def startswith(self, word):
        
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
        return True 

'''
Example
Here 'abcd' is a prefix of 'abcde' and 'bcd' is a prefix of 'bcde'. Since 'abcde' is tested first, print
arr = ['aab','aac','aacghgh','aabghgh'] => o/p = 'BAD SET' aacghgh
arr = ['bcde', 'abcd', 'abcde', 'bcd'] ==> o/p = 'BAD SET' bcd 

'''

def noPrefix(words):
    
    myTrie = Trie()
    myTrie.insert(words[0])
    m = len(words)
    for i in range(1, m):
        if (myTrie.startswith(words[i])):
            print("BAD SET")
            print(words[i]) 
            return        
        elif not myTrie.insert(words[i]):
            print("BAD SET")
            print(words[i]) 
            return
    
    print('GOOD SET')

'''
Find the total  pairs whose difference is 2
https://leetcode.com/problems/k-diff-pairs-in-an-array/submissions/
'''
k = 2
arr1 = [1, 2, 3, 4]
arr2 = [1,5,3,4,2]

# 1st solution using 2 pointer approach
#2nd solution using hashmap

def pairsGood(k, arr):

    m = len(arr)
    l = 0
    r = 1
    arr.sort()
    if m <= 1:
        return 0
    if m == 2:
        if arr[r] - arr[l] == k:
            return 1
        else:
            return 0 

    tot = 0
    while  r<m:
        
        diff = arr[r] - arr[l]
        if  diff < k:
            #increment right
            r+=1
        elif diff > k:
            l+=1
            if (l==r):
                r+=1
        else:
            tot +=1
            r+=1
            #l+=1
            
    return tot


k =663 
#values = [756264, 302249, 796827, 823208, 867638, 242553, 521027, 53259, 744425, 610233, 551174, 959062, 272019, 502864, 870290, 716560, 974152, 977050, 565332, 243216, 811826, 24100, 619063, 883838, 21969, 361329, 702733, 440142, 293781, 291792, 604998, 858258, 549448, 759257, 964136, 285995, 394838, 709678, 362756, 830378, 715897, 400030, 5959, 899408, 292455, 868964, 904405, 143619, 522353, 720495, 935237, 244542, 52322, 507289, 41680, 15013, 208351, 516229, 583746, 910078, 985454, 536484, 356276, 982654, 811163, 235376, 210340, 906882, 902416, 106783, 711004, 569122, 552500, 721821, 96730, 976387, 721158, 28402, 281425, 108919, 398568, 526883, 52353, 399231, 792105, 15676, 192023, 42343, 715234, 173432, 869627, 41017, 54974, 194095, 792430, 526608, 827820, 22632, 19706, 583449, 411927, 901753, 209014, 52985, 358677, 109622, 812127, 551837, 609597, 942729, 986477, 827157, 386915, 497635, 526911, 743762, 566658, 395501, 412937, 949177, 500875, 194675, 903079, 892517, 884592, 53982, 27739, 983164, 396164, 34477, 438816, 693447, 866089, 443431, 23437, 829715, 197130, 21306, 16339, 745088, 810500, 669273, 760418, 973489, 348268, 225495, 458009, 476376, 193349, 961174, 121123, 934574, 875729, 5296, 68443, 49783, 353086, 868919, 992155, 95603, 702070, 828483, 243879, 484261, 462137, 787806, 134789, 332953, 228453, 442768, 796690, 829146, 987803, 979364, 520364, 53648, 626774, 795334, 231145, 826494, 7285, 285332, 933911, 903742, 975724, 565995, 397905, 341389, 442105, 108734, 110488, 834410, 821882, 396579, 360666, 132107, 199799, 359340, 335183, 700970, 179744, 501538, 29065, 704722, 438153, 553826, 831041, 477482, 566945, 617074, 170086, 441442, 618400, 975478, 324956, 387249, 192686, 963473, 525945, 348221, 710341, 779979, 393834, 172664, 717886, 723024, 617737, 399763, 933248, 788521, 439479, 719169, 976141, 501468, 704059, 22774, 829052, 319065, 130376, 209677, 461474, 91227, 962147, 822545, 17496, 987140, 553163, 297528, 54311]
values = [5296, 5959, 7285, 15013, 15676, 16339, 17496, 19706, 21306, 21969, 22632, 22774]
arr3 = [1, 2, 3, 4, 3]
k =1
#print(pairsGood(2, arr2))
print(pairsGood(663,values))

def findPairs(nums, k: int) -> int:
        
        setm = set(nums) 
        totPairs = 0

        # if k == 0 we need to take care of this special case
        # if hashm has more than 1 of an item then inc count and return
        hashm = dict()
        for item in nums:
            hashm[item] = 1+ hashm.get(item, 0)            

        for val in setm:
            if k == 0:
                if hashm[val] > 1:
                    totPairs+=1
            else:
                if val + k in setm:
                    totPairs +=1

        return totPairs

'''
# make the first quadrants sum of a 2*2 matrix the biggest compared to all others
#  1 2      1 4     4 1
#  3 4 ==>  3 2 ==> 3 2

112 42  83  119             119 114 42 112
56  125 56  49              56  125 101 49
15  78  101 43      ==>     15  78  56  43
62  98  114  108            62  98  83  108

given the initial configuraations , reverse the rows and columns of eaach matrix in the
best possible way so that the sum of elements in the matrix upper-left corner is maximal.
'''
def flippingMatrix(matrix):
    # Write your code here
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    n = rows
    totsum = 0
    for i in range(rows//2):
        for j in range(cols//2):
            
            print(matrix[i][j], matrix[i][n-j-1])
            print(max(matrix[i][j], matrix[i][n-j-1]))
            print(max(matrix[n-i-1][j], matrix[n-i-1][n-j-1]))            
            totsum += max(max(matrix[i][j], matrix[i][n-j-1]), max(matrix[n-i-1][j], matrix[n-i-1] [n-j-1]))
            print(totsum)
   
    return totsum

arr = [1,1,1,2,2,3,3,3]

import heapq

def topkfreq(arr,k):

    hashm = dict()

    for item in arr:
        hashm[item] = 1+hashm.get(item,0)
    
    arr2 = [(-val,key) for key,val in hashm.items()]
    print(arr2)

    heapq.heapify(arr2)
    i = 0
    res = []
    while i < k:
        val, item = heapq.heappop(arr2)
        res.append(item)
        i +=1
    
    return res

print(topkfreq(arr, 2))

def topkfreq2(arr,k):

    hashm = dict()

    for item in arr:
        hashm[item] = 1+hashm.get(item,0)
    
    #arr2 = [(val,key) for key,val in hashm.items()]
    #print(arr2)

    res = []
    i = 0
    for key,val in sorted(hashm.items(), key=lambda x:x[1], reverse=True ):
        if i < k:
            res.append(key)
            i+=1
        else:
            break
    return res
print(topkfreq2(arr, 2))


num = 5

#app 1
def totNoOf1inaNum(num):

    one_sum = 0
    while num:
        one_sum += num & 1
        #num >>= 1
        num = num >> 1
    print(one_sum)        
totNoOf1inaNum(num)

#app 2
# bin(2) ==> 0b10
#bin(5) ==> 0b101
def totNoOf1inaNum2(num):

    one_sum = 0
    bin_rep = bin(num)[2:]
    for i in bin_rep:
        one_sum += int(i)
    print(one_sum)

totNoOf1inaNum2(num)

# 13 = 1101 in bin
# the 2nd bit is set 
def GetIthBit(num, i):

    # left shift 1 by i bits 
    # if i = 2, 0001 => 0010 ==> 0100
    mask = 1 << i
    res = 1 if num & mask != 0 else 0
    print(res)

GetIthBit(13, 2)

def SetIthBit(num, i):

    # left shift 1 by i bits 
    # if i = 2, 0001 => 0010 ==> 0100
    mask = 1 << i
    res = num | mask 
    print(res)


def ClearIthBit(num, i):

    # left shift 1 by i bits 
    # if i = 2, 0001 => 0010 ==> 0100
    mask = ~ (1 << i) # 1011 = inverse of 0100
    res = num & mask # 0101 => 0101 & 1011 => 0001
    print(res)


# find the unique no
nums =[1,2,2,3,1]

def uni(nums):

    ans = 0
    for i in range(len(nums)):

        ans ^= nums[i]

    return ans

print(uni(nums))

# https://www.youtube.com/watch?v=GDFVTZ-kKl0&list=PL5tcWHG-UPH1K7oTJgIbWy6rCMc8-8Lfm&index=7
#String Processing in Python: Spreadsheet Encoding
# 
#A = 1, B =2 ... Z = 26
# AA => A * 26^1 + A * 26^0 = 27
# ord('A') = 65 ord ('B') = 66 ...
# 1*26 + 1*2 = 28
def spreadsheet_encode_column(col_str):
    num = 0
    count = len(col_str)-1
    for s in col_str:
        num += 26**count * (ord(s) - ord('A') + 1)
        count -= 1
    return num


print(spreadsheet_encode_column("ZZ"))
print(spreadsheet_encode_column("AB"))

import collections
def bfs(n, m, edges, s):
    # Write your code here
    
    graph = {i:[] for i in range(n+1)}
    
    for v1,v2 in edges:
        
        graph[v1].append(v2)
        graph[v2].append(v1)
    
    print(graph)
    res = [-1 for i in range(n+1)]
    
    visited = set()
    visited.add(s)
    q = collections.deque()
    q.append([s,0])
    res[s] = 0
    
    while q:
        
        poppedV, dis = q.popleft()
        
        for nei in graph[poppedV]:
            if nei not in visited:
                visited.add(nei)
                q.append([nei,dis+6])
                res[nei] = dis+6
    
    
    res.remove(0)
    print('remove the 0 eement and return from 1')
    print(res[1:])
    return res[1:]

n = 4
m = 2
edges = [[1,2], [1,3]]
# s is start node
print(bfs(n, m , edges, s=1))


strs = 'ababcbbab'
sub = 'ab'

def findsubstr2(strs, sub):

    res = []
    starti = 0
    i = 0
    while i<len(strs):

        j = 0
        starti = i
        found = False
        while j < len(sub) and starti < len(strs):

            if strs[starti] == sub[j]:
                  starti +=1
                  j+=1
                  found = True
            else:
                found = False
                break

        if found == True:
            res.append(i)
            i = starti
        else:
            i+=1
    
    print(res)

print("Finding substr")        
findsubstr2(strs, sub)

# return the length of continues substr in above strs 

def findsubstr(strs, sub):

    res = []
    starti = 0
    while strs:
        indexval = strs.find(sub, starti)
        print(indexval)
        if indexval == -1:
            break
        res.append(indexval)
        #print(res)
        starti = indexval + len(sub)
    
    print(res)
    maxcnt = float('-inf')
    i = 1
    cnt = 1
    
    for i in range(1, len(res)):

        diff = res[i] - res[i-1]
        if diff == len(sub):
            cnt+=1
            maxcnt = max(maxcnt, cnt)
        else:
            cnt = 1

    print('got maxcnt as ',maxcnt)


findsubstr(strs, sub)

arr = [0,2,6,8,10]
def findIndex(res):
    
    maxcnt = float('-inf')
    cnt =1
    for i in range(1, len(res)):

        diff = res[i] - res[i-1]
        if diff == 2:
          cnt+=1 
          maxcnt = max(maxcnt, cnt)
        else:
            cnt = 1 
    print('my cnt is ',maxcnt)

findIndex(arr)


'''
problem 2

op = ['push', 'push', 'push', 'pop' ]
arr = [4, 2, 1, 6]
arr2 = [1, 2, 3, 1]
https://www.geeksforgeeks.org/design-a-queue-data-structure-to-get-minimum-or-maximum-in-o1-time/  
in result put the product of min and max in the current queue
'''
import collections
arr = [4, 2, 1, 6, 3]
arr2 = [1, 2, 3, 1, 4]
operations = ['push', 'push', 'push', 'pop', 'push', 'pop', 'push']
def getMinMaxQueue(operations, arr):

    q = collections.deque()
    minq = collections.deque()
    maxq = collections.deque()

    m = len(operations)
    j = 0
    res = []
    for i in range(m):

        if operations[i] == 'push':
            q.append(arr[j])          

            if i == 0:
                minq.append(arr[j])
                maxq.append(arr[j])
            else:
                
                while minq and minq[-1] >= arr[j]:
                    minq.pop()
                minq.append(arr[j])

                while maxq and maxq[-1] <= arr[j]:
                    maxq.pop()
                maxq.append(arr[j])
                print('min q',minq)
                print('max q',maxq)
            j+=1
            res.append(minq[0] * maxq[0])
            
        else: # pop
            
            val = q.popleft()
            
            if val == maxq[0]:
                maxq.popleft()
            if val == minq[0]:
                minq.popleft()
            
            res.append(minq[0] * maxq[0])


    print(res)
            
getMinMaxQueue(operations, arr2)

#https://www.hackerrank.com/challenges/migratory-birds/problem?utm_campaign=challenge-recommendation&utm_medium=email&utm_source=24-hour-campaign

'''
Given an array of bird sightings where every element represents a bird type id, 
determine the id of the most frequently sighted type. If more than 1 type has been 
spotted that maximum amount, return the smallest of their ids.
arr = [1,1,2,2,3]
There are two each of types 1 and 2, and one sighting of type 3. 
Pick the lower of the two types seen twice: type 1. 
'''
def migratoryBirds(arr):
    # Write your code here
    
    m = len(arr)
    bucl = [0 for i in range(m+1)] 
    for i in range(m):
        bucl[arr[i]] +=1
    
    res = [0]
    maxc = 0
    for i in range(len(bucl)):
        #maxc = max(maxc, bucl[i])
        if maxc < bucl[i]:
            maxc = bucl[i]
            res = []
            res.append(i)
        if maxc == bucl[i]:
            res.append(i)
    
    return res[0]

# https://www.geeksforgeeks.org/print-next-greater-number-q-queries/
arr = [3, 4, 2, 7, 5, 8, 10, 6]

def nextGr(arr):

    res = [-1 for i in range(len(arr))]
    mstack = [0]

    for i in range(1, len(arr)):
        print(mstack)
        print(res)
        while mstack and arr[mstack[-1]] <= arr[i]:
            index = mstack.pop()
            res[index] = arr[i]
        mstack.append(i)

    print(res)

nextGr(arr)


# https://www.geeksforgeeks.org/next-greater-frequency-element/?ref=lbp

'''
Next Greater Frequency Element

    Difficulty Level : Medium
    Last Updated : 20 Jul, 2022

Given an array, for each element find the value of the nearest element to 
the right which is having a frequency greater than as that of the current element.
 If there does not exist an answer for a position, then make the value -1
Input : a[] = [1, 1, 2, 3, 4, 2, 1] 
Output : [-1, -1, 1, 2, 2, 1, -1]
Explanation:
Given array a[] = [1, 1, 2, 3, 4, 2, 1] 
Frequency of each element is: 3, 3, 2, 1, 1, 2, 3
'''
arr = [1, 1, 2, 3, 4, 2, 1]
def nextGreaterFreq(arr):

    hashm = {}
    for i in range(len(arr)):
        hashm[arr[i]] = 1 + hashm.get(arr[i], 0)

    print(hashm)

    stack = [[arr[-1], hashm[arr[-1]]]]
    print('initial stack',stack) 
    res = [0] * len(arr)
    res[-1] = -1

    for i in range(len(arr)-1)[::-1]:

        currfreq = hashm[arr[i]]
        
        # pop from stack when stack freq is less 
        while stack and stack[-1][1] <= currfreq:
            stack.pop()

        # if stack freq is greater than put the stack elem in res and add new ele to stack
        if stack and stack[-1][1] > currfreq:
            res[i] = stack[-1][0]
        else:
            res[i] = -1
        
        stack.append([arr[i],hashm[arr[i]]])
        print(stack)

    print("result ",res)

nextGreaterFreq(arr)

def migratoryBirds(arr):
    # Write your code here
    maxCt = 0
    maxId = []
    hashm = dict()
    arr.sort()
    for i in arr:
        hashm[i] = 1 + hashm.get(i, 0)
        if hashm[i] > maxCt:
            maxCt = hashm[i]
            maxId = i 
        
    print(hashm)
    
    return maxId


class MyAttributeClass(object):
    def __init__(self, **kwargs) -> None:
        i=0
        
        
        for key, val in kwargs.items():
            self.__dict__[key] = val
            i+=1
        self.attr = i

    def __len__(self) -> int:
        return self.attr


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

#import collections
def rottenOr(arr):

    rows = len(arr)
    cols = len(arr[0])
    fresh = 0
    q = collections.deque()
    for i in range(rows):
        for j in range(cols):
            if arr[i][j] == 2:
                q.append([i,j])
            if arr[i][j] == 1:
                fresh +=1

    print(q)
    cnt = 0
    direction = [[1,0],[-1,0],[0,1],[0,-1]]
    while q:

        qsize = len(q)
        while qsize and fresh > 0:

            row,col  = q.popleft()
            
            for di,dj in direction:
                
                drow,dcol = row + di, col + dj
                if (0 <= drow < rows and 0 <= dcol < cols and arr[drow][dcol] == 1):
                    arr[drow][dcol] = 2
                    fresh -=1
                    q.append([drow,dcol])
            qsize-=1
        
        cnt+=1
        if fresh == 0:
            break
    return cnt

val = rottenOr(arr)        
print('rotten oranges are', val)

print('printing bin , oct, hex nums to dec')
print(int(1010))
print('printing binary')
print(int(0b1010))
print(int('1010', 2))
print(int('0b1010', 2))

print('printing octal')
print(int(0o100))
print(int('100', 8))
print(int(0x8E))
print(int('8E', 16))

print('printing dec to  bin , oct, hex nums')
#print(help(bin))
print(bin(10))

print(oct(142))
print(oct(100))

print(hex(100))

'''
Firt Convert the bin num to dec. and then dec to the required base   
'''
def binary_conv(binary, base):

    val = 0
    binary = ''.join(binary)
    
    if base == 'b':
        return binary
    elif base == 'o':
        decint = int(binary, 2)
        octv = oct(decint)
        octv = str(octv)
        return octv
    elif base == 'x':
        decint = int(binary, 2)
        hexv = hex(decint)
        hexv = str(hexv)
        return hexv
    else: # base = d
        decint = int(binary, 2)
        valstr = str(decint)
        return valstr

binary = "010101"
print('printing conv of bin nnumbers')
print(binary_conv(binary, 'x'))

#app 2
# bin(2) ==> 0b10
#bin(5) ==> 0b101
def totNoOf1inaNum2(num):

    one_sum = 0
    bin_rep = bin(num)[2:]
    for i in bin_rep:
        one_sum += int(i)
    print(one_sum)

totNoOf1inaNum2(5)

def decToBin(num):

    stackm = []
    while num :

        rem = num % 2
        num = num // 2
        
        stackm.append(rem)
    
    print(stackm[::-1])

decToBin(75)

print(str(1010))

def binToDec(binno):

    m = len(binno)
    power = m-1

    decVal = 0
    for i in range(m):

        decVal += int(binno[i]) * 2 ** power
        power -=1 
    
    print(decVal)

binToDec("1001011")

print(479//16)
print(479%16)