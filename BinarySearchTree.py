
from logging import NullHandler


class Node:

    def __init__(self, data):

        self.data = data
        self.left = None
        self.right = None

class BST:

    def __init__(self):

        self.root = None
        
    # add a node to a BST
    def insert_helper(self, curr_node, data):
        if curr_node:
            if data < curr_node.data:
                #insert to the left
                if curr_node.left:
                    self.insert_helper(curr_node.left, data)
                    # OR in while loop == curr_node = curr_node.left
                else:
                    curr_node.left = Node(data)
                    print("inserting ", data)
                
            elif data > curr_node.data:
                #insert to the right
                if(curr_node.right):
                    self.insert_helper(curr_node.right, data)
                else:
                    curr_node.right = Node(data)
                    print("inserting ", data)
            else:
                print("data is already present. no dups allowed")

    
    def insert(self, data):
        
        if not self.root:
            self.root = Node(data)
        else:
            self.insert_helper(self.root, data)

    #inorder traversal validates the bst property of the tree
    def print_tree_inorder(self):
        if(self.root):
            trav =[]
            res = True
            res = self.print_tree_inorder_helper(self.root, trav, res)
            print(trav)
            if res == False:
                return False
            return True
    
    def print_tree_inorder_helper(self, node, trav, res):
        if not res:
            print(res)
            return False

        if(node):            
            self.print_tree_inorder_helper(node.left, trav, res)
            print(str(node.data))
            if len(trav) > 1:
                if node.data < trav[-1]:
                    print(" not a BST", node.data, trav[-1])
                    res = False
                    #return False
            trav.append(node.data)
            self.print_tree_inorder_helper(node.right, trav, res)


    def search_val(self, val):
        if self.root:
            if self.search_val_helper(self.root, val):
                return True
            return False
            


    def search_val_helper(self, curr_node, data):
        if curr_node:
            
            if data == curr_node.data:                
                return True
            elif data < curr_node.data:
                return self.search_val_helper(curr_node.left, data)
            else:
                return self.search_val_helper(curr_node.right, data)

    def checkbst(self):

        min = 0
        max = float('inf')

        node = self.root        
        val = self.checkbsthelper(node, min, max)        
        if val == False:
            return False
        return True
    
    def checkbsthelper(self, node, min, max):

        if node is None:
            return True
        if not ( node.data > min and node.data < max ):
            return False
        return (self.checkbsthelper(node.left, min, node.data) and self.checkbsthelper(node.right, node.data, max)) 


    #Function to return the diameter of a Binary Tree.
    def diameter(self):
        # Code here
        res = [0]
        def ht(node):
            
            if not node:
                return 0
            
            leftHt = ht(node.left)
            rightHt = ht(node.right)
            res[0] = max(res[0], leftHt + rightHt)
            
            return 1 + max(leftHt, rightHt)
            
        ht(self.root)
        return res[0]
    
   # Given a binary tree, find if it is height balanced or not. 
   # A tree is height balanced if difference between heights of left and right subtrees is not more than one for all nodes of tree. 
    def isBalanced(self,root):
    
        def isBalancedHelp(node):
        
        #add code here
            if not node:
                return [True,0]
            
            lH = isBalancedHelp(node.left)
            rH = isBalancedHelp(node.right)
            balanced = lH[0] and rH[0] and abs(lH[1] - rH[1]) <= 1 
            
            
            return [balanced, 1+max(lH[1], rH[1])]
        
        return isBalancedHelp(root)[0] 

        
    #Function to convert a binary tree into its mirror tree.
    def mirror(self,root):
        # Code here
        
        if not root:
            return 
        
        self.mirror(root.left)
        self.mirror(root.right)
        
        root.left, root.right = root.right, root.left
    
    def find(self, root, val):

        if not root:
            return False
        else:
            lV = self.find(root.left, val)
            rV = self.find(root.right, val)
            return root.data == val or lV or rV 

    def find2(self, root, val):

        if not root:
            return False
        return root.data == val or self.find2(root.left,val) or self.find2(root.right, val)


    def isSumTree(self,root):
        # Code here
        
        def getSum(root):
            
            if not root:
                return 0
            lr = getSum(root.left)
            rr = getSum(root.right)
            
            return root.data + lr + rr
           
        
        if not root:
            return True
        if not root.left and not root.right:
            return True
        lsum = getSum(root.left)
        rsum = getSum(root.right)
        total = lsum + rsum
        if (root.data == total):
            return self.isSumTree(root.left) and self.isSumTree(root.right)
        else:
            return False

    def isSumTree2(self,root):
        # Code here
        isSum = [1]
        def findisSum(root):

            if not root:
                return 0
            if not root.left and not root.right:
                return root.data
            if not isSum[0]:
                return 0 

            lSum = findisSum(root.left)
            rSum = findisSum(root.right)
            isSum[0] = isSum[0] and (root.data == lSum + rSum)

            return root.data+lSum+rSum

        
        findisSum(root)
        return True if isSum[0] else False

    def findSum(self,root):

        if not root:
            return 0
        lsum = self.findSum(root.left)
        rsum = self.findSum(root.right)

        return root.data+lsum+rsum

    
    #Function to check whether all nodes of a tree have the value 
    #equal to the sum of their child nodes.
    def isSumProperty(self, root):
        # code here
        
        def isSumProperty(root):
            
            if root is None:
                return True
            if not root.left and not root.right:
                return True
            
            if root.left:
                suml =  root.left.data
            else:
                suml = 0
            
            if root.right:
                sumr =  root.right.data
            else:
                sumr = 0
            
            return root.data == suml+sumr and isSumProperty(root.left) and isSumProperty(root.right)
        
        val = isSumProperty(root)
        if val == True:
            return 1
        else: return 0


    def Ancestors(self, root,target):
        '''
        :param root: root of the given tree.
        :return: None, print the space separated post ancestors of given target., don't print new line
        '''
        #code here
        
        def Ances(root,target, path):
            
            if not root:
                return False
            path.append(root.data)
            if root.data == target or Ances(root.left,target, path) or Ances(root.right,target, path):
                return True
            else:
                path.pop()
                return False
        
        path = []
        Ances(root,target, path)
        path.pop()
        path = path[::-1]
        return path

    # Given a Binary Tree. Find the difference between the sum of node values at 
    # even levels and the sum of node values at the odd levels.
    '''
    input:
            1
          /   \
         2     3

    Output: -4

    Explanation:
    sum at odd levels - sum at even levels
    = (1)-(2+3) = 1-5 = -4
    '''
    def getLevelDiff(self, root):
        # Code here
        
        if not root:
            return 0
        
        odd = True
        stack1 = [root]
        stack2= []
        res = []
        
        while stack1 or stack2:
            while stack1:
                popped = stack1.pop()
                res.append(popped.data)
                if popped.left:
                    stack2.append(popped.left)
                if popped.right:
                    stack2.append(popped.right)
            
            while stack2:
                popped = stack2.pop()
                res.append(-popped.data)
                if popped.left:
                    stack1.append(popped.left)
                if popped.right:
                    stack1.append(popped.right)
        
        val=sum(res)
        return val
    

    # return difference of sums of odd
    # level and even level
    def evenOddLevelDifference(root):
    
        if (not root):
            return 0
    
        # create a queue for
        # level order traversal
        q = []
        q.append(root)
    
        level = 0
        evenSum = 0
        oddSum = 0
    
        # traverse until the queue is empty
        while (len(q)):
        
            size = len(q)
            level += 1
    
            # traverse for complete level
            while(size > 0):
            
                temp = q[0] #.front()
                q.pop(0)
    
                # check if level no. is even or
                # odd and accordingly update
                # the evenSum or oddSum
                if(level % 2 == 0):
                    evenSum += temp.data
                else:
                    oddSum += temp.data
            
                # check for left child
                if (temp.left) :
                
                    q.append(temp.left)
                
                # check for right child
                if (temp.right):
                
                    q.append(temp.right)
                
                size -= 1
            
        return (oddSum - evenSum)

    # The main function that returns difference between odd and
    # even level nodes RECUSIVE way
    def getLevelDiff2(self, root):
    
        # Base Case
        if root is None:
            return 0
    
        # Difference for root is root's data - difference for
        # left subtree - difference for right subtree
        return (root.data - self.getLevelDiff2(root.left) - self.getLevelDiff2(root.right))


    
    # Given a sorted array. Convert it into a Height balanced Binary Search Tree (BST).
    # Find the preorder traversal of height balanced BST. If there exist many such balanced BST consider 
    # the tree whose preorder is lexicographically smallest. 
    # Height balanced BST means a binary tree in which the depth of the left subtree 
    # and the right subtree of every node never differ by more than 1.
    # https://www.youtube.com/watch?v=0K0uCMYq5ng
    def sortedArrayToBST(self, nums):
	    # code here
        res = []	            
        def createNode(nums):
	       
            def helper(l , r):
	           
                if (l> r):
                    return None
                mid = (l+r)//2
                root = Node(nums[mid])
                root.left = helper(l,mid-1)
                root.right = helper(mid+1,r)
                return root
                
            return helper(0, len(nums)-1 )	   
	     
        def preordertrav(node):
	       
            if not node:
                return False
            res.append(node.data)
            preordertrav(node.left)
            preordertrav(node.right)
	       
        retnode = createNode(nums)
        preordertrav(retnode)
        return res 
    

   
bst = BST()
bst.insert(7)
bst.insert(9)
bst.insert(3)
bst.insert(5)
bst.insert(11)
bst.insert(6)



print(bst.print_tree_inorder())
print(bst.search_val(3))

bt = BST()
bt.insert(7)
bt.root.left = Node(5)
bt.root.right = Node(11)
bt.root.left.left= Node(4)
bt.root.left.right = Node(6)
bt.root.right.left= Node(3)
bt.root.right.right = Node(12)

#print(bt.print_tree_inorder())
print(bst.checkbst())
print(bt.checkbst())
print(bt.diameter())
print("1st find", bst.find(bst.root, 11))

print("2nd find ",bst.find2(bst.root, 11))

bt1 = BST()
bt1.insert(5)
bt1.root.left = Node(2)
bt1.root.right = Node(3)
bt1.root.left.left= Node(1)
bt1.root.left.right = Node(1)
bt1.root.right.left= Node(2)
bt1.root.right.right= Node(1)

print("total sum of tree",bt1.findSum(bt1.root))
print("printing is sum", bt1.isSumTree2(bt1.root))
#https://www.hackerrank.com/challenges/one-week-preparation-kit-tree-huffman-decoding/problem?isFullScreen=true&h_l=interview&playlist_slugs%5B%5D=preparation-kits&playlist_slugs%5B%5D=one-week-preparation-kit&playlist_slugs%5B%5D=one-week-day-seven
# Enter your code here. Read input from STDIN. Print output to STDOUT
def decodeHuff(root, s):
    #Enter Your Code Here
    sstr = ""
    node = root
    lenstr = len(s)
    i = 0
   
    while i < lenstr and node:
        if s[i] == '0': #move left
            node = node.left
            i+=1
        else: #move right            
            node = node.right
            i+=1
        if node.left is None and node.right is None:
            sstr += node.data
            node = root
    print(sstr)


'''
Diameter of a Binary Tree
Easy Accuracy: 50.01% Submissions: 100k+ Points: 2

The diameter of a tree (sometimes called the width) is the number of nodes on the longest path 
between two end nodes. The diagram below shows two trees each with diameter nine, the leaves that
 form the ends of the longest path are shaded (note that there is more than one path in each tree
  of length nine, but no path longer than nine nodes). ] 
'''

def diameter(self,root):
    # Code here
    res = [0]
    def ht(node):
        
        if not node:
            return -1
        
        leftHt = ht(node.left)
        rightHt = ht(node.right)
        res[0] = max(res[0], leftHt + rightHt + 2)
        
        return 1 + max(leftHt, rightHt)
        
    ht(root)
    return res[0] + 1


#construct BST from preorder trav - amazing
    #https://www.youtube.com/watch?v=UmJT3j26t1I
# 1 naive appraoch is that we can sort the preorder arr and make it inorder where values are sorted
#and we already know how to make a BST from a preorder and inorder traversal - this will take nlogn to sort 
# and n time to create the tree from that data
# 2 approach - create BST just like we we check for bst with upper and lower bounds
# here we wil just need upper bounds.

def buildBSTfromPreorder(arr):

    index = [0]
    maxBound = float('inf')

    def buildhelper(arr, index, maxBound):

        if index[0] == len(arr) or arr[index[0]] > maxBound:
            return None
        root = Node(arr[index[0]])
        index[0] +=1
        root.left = buildhelper(arr, index, root.data)
        root.right = buildhelper(arr, index, maxBound)
        return root


    bst = BST()
    bst.root = buildhelper(arr, index, maxBound)
    return bst

arr = [7,5,4,6,11,10,12]
btnew = buildBSTfromPreorder(arr)
print("printing inoreder...")
btnew.print_tree_inorder()


'''
You are given pointer to the root of the binary search tree and two values
v1 and v2. You need to return the lowest common ancestor (LCA) of v1 and v2 
in the binary search tree. 
https://www.hackerrank.com/challenges/binary-search-tree-lowest-common-ancestor/problem?utm_campaign=challenge-recommendation&utm_medium=email&utm_source=24-hour-campaign
'''

def getpathForVertex(root, data):
    
    path = []
    def getpathhelper(node, data):
        
        if node == None:
            return False     
        path.append(node.data)
        if node.data == data:
            return True
        elif node.data > data:
            return getpathhelper(node.left, data)
        else:            
            return getpathhelper(node.right, data)
                
    
    val = getpathhelper(root, data)
    return path if val == True else []
    

    
def lcam(root, v1, v2):
    #Enter your code here
    
    path1 = getpathForVertex(root, v1)
    path2 = getpathForVertex(root, v2)
    
    i = j = 0
    res = 0
    while i < len(path1) and j < len(path2) and path1[i] == path2[j]:
        i+=1
        j+=1
    res = path1[i-1]
            
    return res 

print('printing lca')
print(lcam(btnew.root, 10, 12))

#inorder successor in BST
#https://www.youtube.com/watch?v=SXKAD2svfmI
# naive approach - traverse the tree in inorder and return the kth position

def inorderSuccessorinBST(root, val):
    successor = None

    while root:

        if root.data <= val:
            root = root.right
        else:
            successor = root
            root = root.left

    return successor

print("print successor node")
succesornode = inorderSuccessorinBST(btnew.root, 6)
if(succesornode):
    print(succesornode.data)

#Flatten a Binary Tree to Linked List
#https://www.youtube.com/watch?v=sWf7k1x9XR4
# the inorder trav should be represented in a linked lis form
'''
            1
        2       5
    3     4   6    7

    1-2-3-4-5-6-7-null
'''

def FlattenBintoLinkedlist(root):

    stack = [root]
    while stack:
        curr = stack.pop()

        if curr.right:
            stack.append(curr.right)
        if curr.left:
            stack.append(curr.left)
        if (len(stack)):
            curr.right = stack[-1]
        curr.left = None

#Insert a given node in a BST
#https://www.youtube.com/watch?v=FiFiNvM29ps

def insertNodetoBST(root, val):
    
    node = Node(val)
    if not root:
        root = node
        return 
    curr = root   

    while True:
        if node.data < curr.data:
            if curr.left:
                curr = curr.left
            else:
                curr.left = node
                break                
        elif node.data > curr.data:
            if curr.right:
                curr = curr.right
            else:
                curr.right = node
                break
        else:
            break

insertNodetoBST(btnew.root, 8)
print("printing inoreder...")
btnew.print_tree_inorder()
print("print boundry trav")


nums = [4, 5, 6, 7, 8, 10, 11, 12]

def sortedArrayToBST(nums):

    def buildarrhelper(l, r):

        if l>r :
            return None
        mid = (l+r)//2
        root = Node(nums[mid])
        root.left = buildarrhelper(l,mid-1)
        root.right = buildarrhelper(mid+1,r)

        return root


    return buildarrhelper(0, len(nums)-1)

def print_inorder(root):

    if not root:
        return 
    print_inorder(root.left)
    print(root.data)
    print_inorder(root.right)

print("clling sorted")
node = sortedArrayToBST(nums)
print_inorder(node)