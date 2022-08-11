class Node:
    def __init__(self, info): 
        self.info = info  
        self.left = None  
        self.right = None 
        

    def __str__(self):
        return str(self.info) 

class BinarySearchTree:
    def __init__(self): 
        self.root = None

    def create(self, val):  
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root
         
            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break

"""
Node is defined as
self.left (the left child of the node)
self.right (the right child of the node)
self.info (the value of the node)
"""
def preOrder(root):
    #Write your code here
    
    trav = []
    def pre_trav(root, trav):
        
        if not root:
            return 
        trav.append(root.info)
        pre_trav(root.left, trav)
        pre_trav(root.right, trav)
    
    pre_trav(root, trav)
    sTrav = " ". join(map(str,trav))
    print(sTrav )

def preOrder2(root):
    #Write your code here
    
    if (root):
        print(root.info,end = " ")
        preOrder2(root.left)
        preOrder2(root.right)



# t = no of nodes
# values = arr = [5 , 8, 1, 6, 9]  
# t = 5 and give input as 5 8 1 6 9
tree = BinarySearchTree()
t = int(input())

arr = list(map(int, input().split()))
if len(arr) < t:
    print("invalid input")
else:     
    for i in range(t):
        tree.create(arr[i])

    preOrder2(tree.root)
    preOrder(tree.root)
