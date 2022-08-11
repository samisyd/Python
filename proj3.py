# add 1 to a number in an array and print

A = [1,5,9]
A2 = [9,9,9]
str1 = 1 + int("".join(map(str, A2)))
print(str1)

'''
sum = A[-1] + 1
carry = 0
if sum == 10 :
        carry = 1
        A[-1] = 0
else:
        A[-1] = sum


'''

def sum_one(A):
    carry = 1
    for i in reversed(range(1, len(A))):
        
        print(i)
        sum = A[i] + carry
        if sum == 10 :
            carry = 1
            A[i] = 0
        else:
            carry = 0
            A[i] = sum
    
    A[0] += carry
    if A[0] == 10:
        A[0] = 1
        A.append(0)
    return A


def sum_one_sol2(A):
    
    A[-1] +=1
    print(A)
    for i in reversed(range(1, len(A))):
        if A[i] != 10:
            break
        A[i] = 0
        A[i-1] +=1

    if A[0] == 10:
        A[0] = 1
        A.append(0)
    
    return A
    
print(sum_one_sol2(A))
print(sum_one_sol2(A2))

for i in range(10,0,-1):
    print(i)

print('new')
print('')
for i in reversed(range(1,10)):
    print(i)


#https://www.youtube.com/watch?v=0sWShKIJoo4
def longestCommonPrefix(strs):
        
        lcp = strs[0]

        for j in range(1, len(strs)):
            i = 0
            substr = ""   
            while i < len(lcp) and i < len(strs[j]) and lcp[i] == strs[j][i]: 
                
                substr += lcp[i]
                i+=1
            if i == 0:
                return ""

            #lcp = lcp[0:i]
            lcp = substr

        return lcp  



def longestCommonPrefix2(strs):

    res = ""

    for i in range(len(strs[0])):

        for item in strs:
            if i == len(item) or strs[0][i] != item[i]:
                return res
        res += (strs[0][i])

    return res 
            


strs = ["flower","flow","flight"]
print(longestCommonPrefix(strs))
print(longestCommonPrefix2(strs))

# Remove Duplicates from Sorted Array -

arr = [0,0,1,1,1,2,2,3,3,4] 

def removeDup(arr):

    l = 0
    r = 1

    while l < r and r <len(arr):

        if arr[l] == arr[r]:
            arr[l] = arr[r]
            r+=1
        else:
            l +=1
            arr[l] = arr[r]
            r+=1
    
    l+=1
    while l < len(arr):        
        arr[l] = '_'
        l+=1
    
    return arr

print(removeDup(arr))

def removeDup1(arr):

    l =1

    for r in range(1, len(arr)):

        if arr[r] != arr[r-1]:
            arr[l] = arr[r]
            l+=1

    return l

'''
Rotten oranges
Goal : Find in how much time can all oraanges rot
2 - Rotten, 1 - Fresh, 0 - empty space
'''
import collections
arr = [ 
        [2, 1, 0, 2],
        [1, 0, 1, 2],
        [1, 0, 0, 1]
    ]

def rottenOr(arr):

    rows = len(arr)
    cols = len(arr[0])

    mqueue = collections.deque()
    rotArr = []
    freshOr = 0
    for i in range(rows):
        for j in range(cols):
            if arr[i][j] == 2:
                rotArr.append([i,j])
            elif arr[i][j] == 1:
                freshOr +=1
    
    visited = set()
    mqueue = collections.deque(rotArr)
    directions = [[1,0],[-1,0],[0,1],[0,-1]]
    count = 0

    while mqueue:

        qsize = len(mqueue)
        print(mqueue)
        
        for _ in range(qsize):
            x, y = mqueue.popleft()
            visited.add((x,y))

            for dirR,dirC in directions:

                newx , newy = x + dirR, y+dirC
                if 0 <= newx < rows and 0 <= newy < cols and arr[newx][newy] == 1:
                    arr[newx][newy] = 2
                    freshOr -=1
                    if (newx,newy) not in visited:
                        mqueue.append([newx,newy])
        
        count +=1
        print("increment count", count, freshOr)
        if freshOr == 0:
            return count

    return count

print("checking rotten")
print(rottenOr(arr))