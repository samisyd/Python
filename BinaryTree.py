import collections


class Queue:

    def __init__(self):
        self.queue = []

    def enqueue(self,data):
        self.queue.insert(0,data)

    def dequeue(self):
        return self.queue.pop()

    def isEmpty(self):
        return self.size() == 0

    def size(self):
        return len(self.queue)


class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinaryTree:

    def __init__(self, data):
        self.root = Node(data)

    def print_traversal(self, travel_type):

        if(travel_type == "pre"):
            return self.preorder_traversal(self.root, "")
        elif travel_type == "post":
            return self.postorder_traversal(self.root, "")
        elif travel_type == "level":
            return self.level_order_BFT(self.root)
        elif travel_type == "reverse":
            return self.reverse_level_order(self.root)
        
        else:
            print("Travel type not supported")
            return False

    def level_order_BFT(self, start):

        if(start == None): return

        queue = Queue()
        queue.enqueue(start)
        traversal = ""
        while not queue.isEmpty():

            node = queue.dequeue()
            print(queue)
            traversal = traversal + "-" + str(node.data)
            if node.left:
                queue.enqueue(node.left)
            if node.right:
                queue.enqueue(node.right)

        return traversal

    def reverse_level_order(self, start):

        if start == None: return

        queue = Queue()
        queue.enqueue(start)
        stack1 = []

        while not queue.isEmpty():
            
            node = queue.dequeue()
            stack1.append(node.data)

            if(node.right):
                queue.enqueue(node.right)
            if(node.left):
                queue.enqueue(node.left)

        traversal = ""   
        while len(stack1):
            #print(stack1.pop())
            traversal = traversal + "-" + str(stack1.pop())
        return traversal

    def preorder_traversal(self, node, traversal):

        if(node):
            traversal = traversal + '-' + str(node.data)
            traversal = self.preorder_traversal(node.left, traversal)
            traversal = self.preorder_traversal(node.right, traversal)
        return traversal

    def postorder_traversal(self, node, traversal):

        if(node):
            traversal = self.postorder_traversal(node.left, traversal)
            traversal = self.postorder_traversal(node.right, traversal)
            traversal = traversal + '-' + str(node.data)
        return traversal  

    def height(self, node):
        
        if not node:
            return 0
        height_left = self.height(node.left)
        height_right = self.height(node.right)

        return 1+ max(height_left, height_right)


    #inorder traversal validates the bst property of the tree
    def print_tree_inorder(self):
        trav =[]
        if(self.root):
            self.print_tree_inorder_helper(self.root, trav)
            print(trav)            
    
    def print_tree_inorder_helper(self, node, trav):
        if(node):       
            self.print_tree_inorder_helper(node.left, trav)
            trav.append(node.data)
            self.print_tree_inorder_helper(node.right, trav)

    
    def isHeap(self, root):
        #Code Here        
        if root == None:
            return True

        queue = collections.deque([root])   
        emptynode = False
        while queue:
            
            node = queue.popleft()
           
            if (node.left != None) :
                
                if not emptynode and node.left.data > node.data:
                    queue.append(node.left)
                else:
                    return False
            else:
                emptynode = True
                
            if (node.right != None) :
                
                if not emptynode and node.right.data > node.data:
                    queue.append(node.right)
                else:
                    return False
            else:
                emptynode = True
                
        return True

    #Function to check whether a Binary Tree is BST or not.
    def isBST(self, root):        
        #code here
        
        def isBSThelper(node, minVal, maxVal):
            
            if not node:
                return True
            
            if node.data < minVal or node.data > maxVal:
                return False
            else:
                return isBSThelper(node.left, minVal, node.data) and isBSThelper(node.right, node.data, maxVal)     
        
        return isBSThelper(root, -1, 100000)        

    #DFS - preordertraversal
    def DFStree(self, root):

        if not root:
            return 
        
        print(root.data)
        self.DFStree(root.left)
        self.DFStree(root.right)    

    #BFS - level order traversal
    def BFStree(self, root):

        queue = []
        if not root:
            return
        
        i=0
        queue.append(root)
        while i < len(queue):

            popped = queue[i]
            i+=1
            print(popped.data)
            if popped.left:  queue.append(popped.left)
            if popped.right: queue.append(popped.right)
            

    # find path in tree
    def DFSPathHelpertree(self, root, destVal, path):

            if not root:
                return False            
            print(root.data)
            path.append(root.data)
            print(path)
            if root.data == destVal or self.DFSPathHelpertree(root.left, destVal, path) or self.DFSPathHelpertree(root.right, destVal, path):
                return True
            else:
                path.pop()
                return False

    def DFSFindPathtree(self, dest):
        path = []
        self.DFSPathHelpertree(self.root, dest, path)
        return path

    # find kth parent
    def findkParent(self, kthParent, val):

        path = self.DFSFindPathtree(val)
        if kthParent >= len(path): 
            return False
        else:
            return path[len(path) - kthParent - 1]

    #find LCA - lowest common ancestor
    def lcaTree(self, val1, val2):

        path1 = self.DFSFindPathtree(val1)
        path2 = self.DFSFindPathtree(val2)

        i = j =0            
        lcaval = None
        while i < len(path1) and j < len(path2):
            
            if path1[i] == path2[j]:
                lcaval = path1[i]
                i +=1
                j +=1
            else:
                break
        return lcaval
    
    # find all paths till leaf
    def findAllPathstillLeaf(self):

        path = []
        paths = []
        node = self.root
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

# https://www.youtube.com/watch?v=d4zLyf32e3I bin tree right side view
    def rightSideView(self, root):

        res = []
        q = collections.deque([root])        

        while q:
            rightside = None
            qlen = len(q)
            
            for i in range(qlen):
                node = q.popleft()
                
                if node:                    
                    rightside = node.data
                    if(node.left):
                        q.append(node.left)
                    if(node.right):
                        q.append(node.right)
                        

            res.append(rightside)
        
        return res

    # https://www.youtube.com/watch?v=KV4mRzTjlAk
    def rightView(self,root):
        
        level = 0
        res = []
        
        def rightViewhelper(node, level, res):

            if node is None:
                return None

            if level == len(res):
                res.append(node.data)

            rightViewhelper(node.right, level+1, res)
            rightViewhelper(node.left, level+1, res)

        rightViewhelper(root, level, res)
        return res

    def leftView(self,root):
        
        level = 0
        res = []
        
        def rightViewhelper(node, level, res):

            if node is None:
                return None

            if level == len(res):
                res.append(node.data)

            rightViewhelper(node.left, level+1, res)
            rightViewhelper(node.right, level+1, res)

        rightViewhelper(root, level, res)
        return res

    def DFST(self, node):

        def DFSThelper(node, path):
            
            if not node:
                return None
            path.append(node.data)
            DFSThelper(node.left, path)
            DFSThelper(node.right, path)
        
        path = []
        DFSThelper(node, path)
        return path

    def BFST(self, node):

        queue = [node]
        path = []
        if node is None:
            return None
        i = 0
        while i < len(queue):
            popped = queue[i]
            i+=1
            path.append(popped.data)
            if popped.left: queue.append(popped.left)
            if popped.right: queue.append(popped.right)  
        
        return path 
       
    
    #DFS - preorder no recursion
    # data- left- right
    def preOrder(self, root):
        # code here
        res = []
        if not root:
            return None
        
        stack = [root]
        
        while len(stack):
            
            popped = stack.pop()
            res.append(popped.data)
            if popped.right:
                stack.append(popped.right)
            if popped.left:
                stack.append(popped.left)
        
        return res

    # inorder - non recursive - left-data-right
    def inOrder(self, root):
        # code here
        stack = []
        res = []
        curr = root
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            res.append(curr.data)
            curr = curr.right
            
        return res
    

    # https://www.youtube.com/watch?v=PiqIXedWhoY
    # preorder is - data left right - we used stack
    # post order is = left right data ?
    # just do reverse of pre-order but change the right to left when inserting
    # do preorder like - data right left - reverse will give - left right and data
    # channel - https://www.youtube.com/channel/UCncHG0XiVpZeHhUY8mGWVZw/videos
    def postOrder(self, node):
        # code here
        res = []
        stack = [node]
        root = node

        # code here
        res = []
        if not root:
            return None
        
        stack = [root]
        
        while len(stack):
            
            popped = stack.pop()
            res.append(popped.data)
            if popped.left:
                stack.append(popped.left)
            if popped.right:
                stack.append(popped.right)
            
        res = res[::-1]

        return res


    '''
    Given a Binary Search Tree. Convert a given BST into a Special Max Heap with the 
    condition that all the values in the left subtree of
    a node should be less than all the values in the right subtree of the node. 
    This condition is applied on all the nodes in the so converted Max Heap.

    Your task :
    You don't need to read input or print anything. Your task is to complete the function 
    convertToMaxHeapUtil() which takes the root of the tree as input and converts the BST to max heap.
    Note : The driver code prints the postorder traversal of the converted BST.

    Input :
                         4
                     /   \
                   2     6
                /  \   /  \
               1   3  5    7  

    Output inorder traversal of BST gives sorted array : 1 2 3 4 5 6 7 
    Exaplanation :
                   7
                /   \
                3     6
               /   \  /   \
               1    2 4     5
    The given BST has been transformed into a
    Max Heap and it's postorder traversal LEFT-RIGHT-DATA is
    1 2 3 4 5 6 7.

    Approach 
    1. Create an array arr[] of size n, where n is the number of nodes in the given BST. 
    2. Perform the inorder traversal of the BST and copy the node values in the arr[] in sorted 
    order. 
    3. Now perform the postorder traversal of the tree. 
    4. While traversing the root during the postorder traversal, one by one copy the values from the array arr[] to the nodes.

    '''
    def convertBSTToMaxHeapUtil(self, root):
        #code here
        
        # get the inorder traversal of the tree in a stack
        def GetInorderArr(root):
            
            stack = []
            def convertToInorderArrhelper(node):
                
                if node is None:
                    return
                if node.left:
                    convertToInorderArrhelper(node.left)
                stack.append(node.data)
                if node.right:
                    convertToInorderArrhelper(node.right)
            
            convertToInorderArrhelper(root)    
            #print(stack)
            return stack

        # superimpose  the inorder stack elements on the tree elements in a post order way
        #  Now perform the postorder traversal of the tree. 
        # While traversing the root during the postorder traversal, one by one copy the values from the array stack[] to the nodes.

        def convertToPostOrder(root, stk):
            
            stack2 = []
            def postOrderhelper(node, stack, val):
                
                if node is None:
                    return
                if node.left:
                    postOrderhelper(node.left, stack, val[0])
                if node.right:
                    postOrderhelper(node.right, stack, val[0])
                
                node.data = stack[val[0]]
                stack2.append(node.data)
                val[0] = val[0]+1
            
            val = [0]
            postOrderhelper(root, stk, val)
            print(stack2)
        
        # get the inorder traversal of the tree in a stack
        stk = GetInorderArr(root)
        # superimpose  the inorder stack elements on the tree elements in a post order way      
        convertToPostOrder(root, stk)
       
    '''
    Diameter of a Binary Tree 
    The diameter of a tree (sometimes called the width) is the number of nodes on the
     longest path between two end nodes. The diagram below shows two trees each with 
     diameter nine, the leaves that form the ends of the longest path are shaded (note 
     that there is more than one path in each tree of length nine, but no path longer
      than nine nodes). 
    '''    

    #Function to return the no of nodes in diameter of a Binary Tree.
    def diameter2(self):
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
        return res[0] + 1



    #Function to check if two trees are identical.
    def isIdentical(self,root1, root2):
        # Code here            
            
        def getPreorderTravHelper(node1, node2):
            
            if not node1 and not node2:
                return True
            if node2 and not node1:
                return False
            if node1 and not node2:
                return False
            
            return node1.data == node2.data and getPreorderTravHelper(node1.left, node2.left) and getPreorderTravHelper(node1.right, node2.right)
                
        val = getPreorderTravHelper(root1, root2)
        if val:
            return 'Yes'
        return 'No'

    #Function to store the zig zag order traversal of tree in a list.
    # Given a Binary Tree. Find the Zig-Zag Level Order Traversal of the Binary Tree.
    # https://www.youtube.com/watch?v=YsLko6sSKh8
    # take 2 stacks , take root in stack1 , pop from stack1 and write o o/p and push children left-to-right in stack2 till stack1 is empty then
    #  pop from stack2 and write to o/p and push children right-to-left in stack1 till stack2 is empty 

    def zigZagTraversal(self, root):
        #Add Your code here
        stack1 = [root]
        stack2 = []
        result = []
        
        while stack1 or stack2:
            
            while stack1:
                node = stack1.pop()
                result.append(node.data)
                
                if node.left:
                    stack2.append(node.left)
                if node.right:
                    stack2.append(node.right)
            
            while stack2:
                node = stack2.pop()
                result.append(node.data)
                
                if node.right:
                    stack1.append(node.right)
                if node.left:
                    stack1.append(node.left)
                    
        return result

    # recursive way
    # Given a Binary search tree. Your task is to complete the function which will return the 
    # Kth largest element without doing any modification in Binary Search Tree
    def kthLargest(self,root, k):
        #your code here
        res=[]
        #doinordertraversal
        def inordertrav(node):
            
            if not node:
                return
            inordertrav(node.left)
            res.append(node.data) 
            inordertrav(node.right)
            
        inordertrav(root)
        #get kth  largests element
        if k > len(res):
            return 0
        else:
            return res[len(res)-k]

    # Given a Binary search tree. Your task is to complete the function which will return the Kth smallest element 
    # without doing any modification in Binary Search Tree
    def kthsmallest(self,root,k):

        i = 0
        curr = root
        stack = []
        res = []

        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left

            curr = stack.pop()
            res.append(curr.data)
            i +=1
            if k==i:
                return curr.data
            
            curr = curr.right

    def find2(self, root, val):

        if not root:
            return False
        return root.data == val or self.find2(root.left,val) or self.find2(root.right, val)

    
    #Function to find the vertical order traversal of Binary Tree.
    '''
    Given a Binary Tree, find the vertical traversal of it starting from the leftmost level to the rightmost level.
    If there are multiple nodes passing through a vertical line, then they should be printed as they appear in level order traversal of the tree.

        Input:
           1
         /   \
       2       3
     /   \   /   \
    4      5 6      7
              \      \
               8      9           
    Output: 
    4 2 1 5 6 3 8 7 9 
    '''
    import collections
    def verticalOrder(self, root): 
        #Your code here
        
        queue = collections.deque([(root,0)])
        hashmap = dict()
        
        while queue:
            
            popped,dis = queue.popleft()
            if dis in hashmap.keys():
                hashmap[dis].append(popped.data)
            else:
                hashmap[dis] = [popped.data]
            
            if popped.left:
                queue.append([popped.left, dis-1])
            if popped.right:
                queue.append([popped.right, dis+1])
        
        
        res = []    
        for key, v in sorted(hashmap.items()):
            
            res.extend(v)
        return res


    def depthTree(self,root):

        if not root:
            return 0

        return 1 + max(self.depthTree(root.left), self.depthTree(root.right)) 

    def depthTreeBFS(self,root):

        queue = collections.deque([root])
        res = 0

        while queue:            
            lenq = len(queue)

            while lenq:
                popped = queue.popleft()

                if popped.left:
                    queue.append(popped.left)
                if popped.left:
                    queue.append(popped.right)
                lenq -=1
            
            res +=1

        return res

    def depthTreeDFS(self,root):

        res = 0
        if not root:
            return res
        stack = [[root,1]]
        
        while stack:
            popped,depth = stack.pop()
            res = max(res, depth)

            if popped.left:
                    stack.append([popped.left, depth+1])
            if popped.right:
                    stack.append([popped.right, depth+1])
        return res

    # check if the tree is a balanced tree. the diff should be 0 or 1 bt not more
    def isBalanced(self,root):

        balanced = [True]
        def isBalancedHelp(node):
            
            #add code here
            if not node:
                return 0
            
            lH = isBalancedHelp(node.left)
            rH = isBalancedHelp(node.right)
            balanced[0] = balanced[0] and (abs(lH - rH) <= 1)            
            return 1+max(lH, rH)
        
        isBalancedHelp(root) 
        return balanced


bt2 = BinaryTree(4)
bt2.root.left = Node(2)
bt2.root.right = Node(6)
bt2.root.left.left= Node(1)
bt2.root.left.right = Node(3)
bt2.root.right.left= Node(5)
bt2.root.right.right = Node(7)

print("paths ==> ",bt2.findAllPathstillLeaf())
print("printing depth recursive",bt2.depthTree(bt2.root))
print("printing depth BFS",bt2.depthTreeBFS(bt2.root))
print("printing depth DFS",bt2.depthTreeDFS(bt2.root))


print(bt2.kthsmallest(bt2.root, 3))
print(bt2.kthLargest(bt2.root, 5))

#print(bt2.isBST(bt2.root))
#bt2.convertBSTToMaxHeapUtil(bt2.root)

bt = BinaryTree(1)
bt.root.left = Node(2)
bt.root.right = Node(3)
bt.root.left.left= Node(4)
bt.root.left.right = Node(5)
bt.root.right.left= Node(6)
bt.root.right.right = Node(7)
bt.root.right.right.right = Node(8)

bt3 = BinaryTree(1)
bt3.root.left = Node(2)
bt3.root.right = Node(3)
bt3.root.left.left= Node(4)
bt3.root.left.right = Node(5)
bt3.root.right.left= Node(6)
bt3.root.right.right = Node(7)
bt3.root.right.right.right = Node(8)
#bt3.root.right.right.right.right = Node(9)
#bt3.getdepthBFS(bt3.root)
#bt3.getdepthDFS(bt3.root)
#print("check is balanced",bt3.isBalanced(bt3.root))

# check for - 628 504 N 505 438 N 70 90
# 5 488 N 230 57 N 281
print("printing isIdentical")
print(bt.isIdentical(bt.root, bt2.root))

print("printing diameter2", bt.diameter2())


print("printing inorder")
bt.print_tree_inorder()

print("printing post order", bt.postOrder(bt.root))
print(bt.DFST(bt.root))
print(bt.BFST(bt.root))
#print(bt.FindallPaths(bt.root))
#print(bt.findAllPathstillLeaf())



#print(bt.reverse_level_order(bt.root))
print("right side view")
#print(bt.rightSideView(bt.root))
print("right side view 2")
print(bt.rightView(bt.root))
print(bt.leftView(bt.root))

#print(bt.print_traversal("pre"))
#print(bt.print_traversal("post"))
#print(bt.print_traversal("level"))
#print(bt.print_traversal("reverse"))
#print(bt.height(bt.root))
#bt.DFStree(bt.root)
path = bt.DFSFindPathtree(4)
print("path is ", path)
#print(bt.findkParent(3, 8))

#print(bt.lcaTree(6, 8))
#print("printing BFS")
#bt.BFStree(bt.root)
#paths = bt.findAllPathstillLeaf()
#print(paths)

bst = BinaryTree(20)
bst.root.left = Node(18)
bst.root.right = Node(19)
bst.root.left.left= Node(14)
bst.root.left.right = Node(10)
bst.root.right.left= Node(21)
bst.root.right.right = Node(6)
#bst.root.right.right.right = Node(4)
#print("Is Heap", bst.isHeap(bt.root))

#print('printing result', bt.findleftViewNew())

import collections
class MyNode:

    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class MyTree:

    def __init__(self, data):
        self.root = Node(data)

    #Function to find the vertical order traversal of Binary Tree.
    '''
    Given a Binary Tree, find the vertical traversal of it starting from the leftmost level to the rightmost level.
    If there are multiple nodes passing through a vertical line, then they should be printed as they appear in level order traversal of the tree.

        Input:
           1
         /   \
       2       3
     /   \   /   \
    4      5 6      7
              \      \
               8      9           
    Output: 
    4 2 1 5 6 3 8 7 9 
    '''
    
    def verticalOrder(self, root):

        # add to queue the node and distance from root
        q = collections.deque()
        q.append([root, 0])
        hashm = dict()
        res = []

        while q:

            node, dis = q.popleft()
            if dis in hashm:
                hashm[dis].append(node.data)
            else:
                hashm[dis] = [node.data]
            
            if node.left:
                q.append([node.left,dis-1])
            if node.right:
                q.append([node.right,dis+1])
            #print(q)

        for key,val in sorted(hashm.items()):
            res.extend(val)

        return res

    
mtree = MyTree(1)
mtree.root.left = Node(2)
mtree.root.right = Node(3)
mtree.root.left.left = Node(4)
mtree.root.left.right = Node(5)
mtree.root.right.left = Node(6)
mtree.root.right.right = Node(7)
mtree.root.right.left.right = Node(8)
mtree.root.right.right.right = Node(9)

res = mtree.verticalOrder(mtree.root)
print(res)