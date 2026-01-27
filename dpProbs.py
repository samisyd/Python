# /*
#  * Great work Striver. Just couple of observations since f(n) denotes no of ways to reach nth 
#  * stair from step 0. It implies f(0) should be 0(you don't have to move at all, you are already 
#  * at your destination), f(1)=1(because only one step of single jump needed) and f(2) = 2 because 
#  * in this case either two times a 1-step can be taken or a single 2-step works. So closely 
#  * this pattern doesn't resemble fibonacci till index 2, because f(2) is not a sum of f(1) and f(0),
#  * rather they are adding upto 1
#  */

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """

        if (n <=2):
         return n
        
        l = self.climbStairs(n-1 )
        r = self.climbStairs(n-2 )
        return l + r
    

    def climbStairsWithMemo(self, n):
        """
        :type n: int
        :rtype: int
        """
        memo = {}        
    
        def helper(n):

            if (n <=2):
                memo[n] = n
                return n
            
            if n in memo:
                return memo[n]

            l = helper(n-1 )
            r = helper(n-2 )
            memo[n] = l + r
            return memo[n]        
        return helper(n)
    
    def rob(self, nums: List[int]) -> int:
        def helper(n, nums):

            if n == 0 :
                return nums[0]
            if n < 0 :
                return 0

            pick = nums[n] + helper(n-2, nums)
            notpick = 0 + helper(n-1, nums)

            return max(pick, notpick)
        
        n = len(nums)
        return helper(n-1, nums)
    
    def robMemo(self, nums: List[int]) -> int:  
        n = len(nums)      
        memo = [-1] * n
        
        def helper(n, nums):
            if n == 0 :
                return nums[0]
            if n < 0 :
                return 0
            
            if(memo[n] != -1):
                return memo[n]
            pick = nums[n] + helper(n-2, nums)
            notpick = 0 + helper(n-1, nums)

            memo[n] = max(pick, notpick)
            return memo[n] 
        
        
        return helper(n-1, nums)
    

    def moveZeros(nums):
        zeropos = 0
        for i in nums:
            if i != 0:
                nums[zeropos] = i
                zeropos += 1
        
        while zeropos < len(nums):
            nums[zeropos] = 0
            zeropos += 1

    def isAnagram(str1, str2):
        if len(str1) != len(str2):
            return False
        
        count ={}
        count2 = {}
        for c in str1:
            count[c] = count.get(c, 0) + 1
        for cc in str2:
            count2 = count2.get(cc, 0) + 1

        if count == count2:
            return True

    from collections import defaultdict 
    # strs = ["eaat","teaa","tan","ate","nat","bat"] #[["bat"],["nat","tan"],["ate","eat","tea"]]
    def groupAnagrams(self, strs):

        def getkey(s):
            keycount = [0 for i in range(26)]
            for ch in s:
                indexch = ord(ch) -ord('a')
                keycount[indexch] +=1            
            return keycount
        
        mmap = defaultdict(list)
        for s in strs:
            key = getkey(s)
            # cannot use 'list' as a dict key (unhashable type: 'list')
            mmap[tuple(key)].append(s)
        
        res = []
        for key,val in mmap.items():
            print(f"key is {key} and val is {val}")
            res.append(val)
        return res
    
from collections import defaultdict

# 1. Using int (default value is 0)
count_dict = defaultdict(int)
count_dict['apple'] += 1  # No KeyError, 'apple' initialized to 0 then +1
print(count_dict)  # Output: defaultdict(<class 'int'>, {'apple': 1})

# 2. Using list (default value is an empty list)
group_dict = defaultdict(list)
group_dict['fruits'].append('apple')  # 'fruits' initialized to [] then appends
print(group_dict)  # Output: defaultdict(<class 'list'>, {'fruits': ['apple']})

s = Solution()
print("Climb stairs", s.climbStairs(5))
print("Climb stairs memo", s.climbStairsWithMemo(5))
print("robMemo ", s.robMemo([1,2,3,1]))

nums = [1,2,5,4,0]
myset = set(nums)
if len(myset) != len(nums):
    print("Duplicates found")


strs = ["eaat","teaa","tan","ate","nat","bat"] #[["bat"],["nat","tan"],["ate","eat","tea"]]
ret = s.groupAnagrams(strs)
print("strs are ", ret)

