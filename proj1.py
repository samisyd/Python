#check if string has unique characters


str1 = 'inique'
str2 = 'uniqe'
str3 = 'element'

def chk_unique(stri):
    return len(stri) == len(set(stri))
        

print(chk_unique(str1))
print(chk_unique(str2))
print(chk_unique(str3))


A= "the cow jumps over the moon."
B= "the moon      jumps over the   cow."

C="driving"
D="drigging"

def is_perm(str1, str2):
    str1 = str1.replace(" ", "")
    str2 = str2.replace(" ", "")
    
    if len(str1) != len(str2): return False

    for ch in str1:
        if ch in str2:
            str2 = str2.replace(ch, "", 1)
        else:
            return False

    return len(str2) == 0

print(is_perm(A, B))
print(is_perm(C,D))

# Return the 2 values whose sum is equal to val 
nums = [3,2,5,4,1, 5]
val = 9

def funcChk(nums, val):

    if(len(nums) < 0):
        return False
    dicti = {}
    for i in range(len(nums)):
        if(nums[i] in dicti):
            return [nums[i], nums[dicti[nums[i]]]]
        else:
            dicti[val - nums[i]] = i 

    return False

print(funcChk(nums, val))




def fizzBuzz(num):

    for i in range(1, num+1):     
        
        if i % 3 == 0 and i % 5 == 0 :
            print("FizzBuzz")
        elif i % 3 == 0 :
            print("Fizz")
        elif i % 5 == 0 :
            print("Buzz")
        else: 
            print(i)
    return 0

print(fizzBuzz(15))

#def swap_words(s, x, y):
 #   return y.join(part.replace(y, x) for part in s.split(x))


def swap_words1(s, x, y):

    str = [part.replace(y, x) for part in s.split(x)]

    print(str)
    y = y.join(str)
    return y 

   

print(swap_words1('apples and avocados and avocados and apples', 'apples', 'avocados'))
#'avocados and apples and apples and avocados'


def swapwords(mystr, firstword, secondword):
    splitstr = mystr.split(" ")

    for i in range(len(splitstr)):
        if splitstr[i] == firstword:
            splitstr[i] = secondword
            i+=1
        if splitstr[i] == secondword:
            splitstr[i] = firstword
            i+=1

    newstr = " ".join(splitstr)
    return newstr


a = 'apples and avocados and avocados and apples'
b = a.replace('apples', '#IamYourFather#').replace('avocados', 'apples').replace('#IamYourFather#', 'avocados')
print(b)
#avocados and apples and apples and avocados


# two sum problem
# find the 2 values whose sum is = target
def twoSum(arr, target):
    
    L = 0
    R = len(arr) - 1
    arr.sort()
    print(arr)

    while L < R:
        sum = arr[L] + arr[R]

        if sum < target:
            L+=1
        elif sum > target:
            R-=1
        else:
            return[arr[L], arr[R]]
    return[]
    
arr = [11,7,2,3,16] # 2,3,7,11,16
print(twoSum(arr, 9))

def threeSum(arr, target):
    
    arr.sort()
    print(arr)

    res = []
    for i, val in enumerate(arr):

        if i > 0 and arr[i] == arr[i-1]: continue

        L = i+1
        R = len(arr) -1

        while L < R:

            sum = val + arr[L] + arr[R]

            if sum < target:
                L+=1
            elif sum > target:
                R-=1
            else:
                res.append([val,arr[L],arr[R]])
                L+=1
                while arr[L] == arr[L-1] and L<R:
                    L+=1
    return res



arr2 = [-1, 0, 1, 2, -1, -4]
print(threeSum(arr2, 0))


print("starting.....")
print("starting.....")
print("starting.....")    
#https://www.youtube.com/watch?v=EYeR-_1NRlQ
def fourSum(arr, target):
    arr.sort()
    print(arr)
    quad = []   
    res = []

    def kSum(k, start, target):
        print("calling ksum")    
        if k != 2:            
            for i in range(start, len(arr)-k+1):
                if i > start and arr[i] == arr[i-1]: continue
                quad.append(arr[i])
                print(quad)
                kSum(k-1, i+1, target-arr[i])
                quad.pop()
            return

        L , R = start , len(arr) -1
        while L<R:
            sum = arr[L] + arr[R]

            if sum < target:
                L+=1
            elif sum > target:
                R-=1
            else:
                res.append(quad + [arr[L],arr[R]])
                L+=1
                while arr[L] == arr[L-1] and L<R:
                    L+=1


    kSum(4, 0, target)
    return res

arr3 = [1, 0, -1, 0, -2, 2]
print("star four sum ", fourSum(arr3, 0))

print(arr3)
qu = [4,5]
res = []
res.append(qu + [arr3[0], arr3[1]] )
qu.extend([1,2])
print(qu)
print(res)

'''
 The easiest way to see why is to look at merge sort. In merge sort, the idea is to divide
  the list in half, sort the two halves, and then merge the two sorted halves into one 
  sorted whole. But how do we sort the two halves? Well, we divide them in half, sort them, 
  and merge the sorted halves...and so on.

  So what's our total time cost? O(nlog base2 n) 
  The log base2 n comes from the number of times we have to cut nnn in half
   to get down to sublists of just 1 element (our base case). The additional n comes 
   from the time cost of merging all n items together each time we merge two sorted sublists. 
'''
def merge_sort(list_to_sort):
    # Base case: lists with fewer than 2 elements are sorted
    if len(list_to_sort) < 2:
        return list_to_sort

    # Step 1: divide the list in half
    # We use integer division, so we'll never get a "half index"
    mid_index = len(list_to_sort) / 2
    left  = list_to_sort[:mid_index]
    right = list_to_sort[mid_index:]

    # Step 2: sort each half
    sorted_left  = merge_sort(left)
    sorted_right = merge_sort(right)

    # Step 3: merge the sorted halves
    sorted_list = []
    current_index_left = 0
    current_index_right = 0

    # sortedLeft's first element comes next
    # if it's less than sortedRight's first
    # element or if sortedRight is exhausted
    while len(sorted_list) < len(left) + len(right):
        if ((current_index_left < len(left)) and
                (current_index_right == len(right) or
                 sorted_left[current_index_left] < sorted_right[current_index_right])):
            sorted_list.append(sorted_left[current_index_left])
            current_index_left += 1
        else:
            sorted_list.append(sorted_right[current_index_right])
            current_index_right += 1
    return sorted_list


my_list     = [3, 4, 6, 10, 11, 15]
alices_list = [1, 5, 8, 12, 14, 19]


def mergeArrList(list1, list2):
    
    m = len(list1)
    n = len(list2)
    res = [0 for i in range(m+n)]
    
    i = j = k = 0

    while i < m and j < n:

        if list1[i] < list2[j]:
            res[k] = list1[i]
            i +=1
        else:
            res[k] = list2[j]
            j +=1
        k +=1
    
    while i < m:
        res[k] = list1[i]
        i +=1
        k+=1

    while j < n:
        res[k] = list2[j]
        j +=1
        k+=1
    
    return res

print(" printing lists", mergeArrList(my_list, alices_list))

'''
K = 4
L1 = [1 2 3 4]
L2 = [0 5 10 15]
L3 = [2 4 8 10]
L4 = [3 9 27 81]

Expected Output: 
0 1 2 2 3 3 4 4 5 8 9 10 10 15 27 81
Output according to this solution: 
0 1 2 2 3 3 4 5 4 8 9 10 10 15 27 81
'''

import heapq
# merge k lists
def merge(lists):

    res = []
    n = len(lists)
    j = 0
    
    # hashmap to get the curr value in a particular list
    listHash = {i:1 for i in range(n)}
    #get the first item and put in heap
    heapv = [[lists[i][0], i] for i in range(n) ]
    heapq.heapify(heapv)

    while len(heapv):

        popped,listNo = heapq.heappop(heapv)
        res.append(popped)
        #get value from listHash
        indexVal = listHash[listNo]
        # if values in a list are over then just keep popping from the queue
        # get the next value from this popped list and add to heap and increment the index of this in the hashmap
        if indexVal < len(lists[listNo]):            
            # get list
            val = lists[listNo][indexVal]
            heapq.heappush(heapv, [val, listNo])
            listHash[listNo] +=1

    print(res)
        

K = 4
L1 = [1, 2, 3, 4]
L2 =[0, 5, 10, 15]
L3 =[2, 4, 8, 10]
L4 =[3, 9]
L = [L1,L2, L3, L4]
merge(L)

