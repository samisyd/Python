# implement LRU Cache

'''
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

    LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    int get(int key) Return the value of the key if the key exists, otherwise return -1.
    void put(int key, int value) Update the value of the key if the key exists. 
    Otherwise, add the key-value pair to the cache. 
    If the number of keys exceeds the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

https://leetcode.com/problems/lru-cache/
https://www.youtube.com/watch?v=7ABFKPK2hD4

create a hashmap to get the values fasts
to keep track of the LRU and MRU i.e which ones to delete we need an doubly linked list 
with 2 pointer, one in the front indicating LRU and one in the back indicating MRU   

the hashmap will have pointers to the node in the linked list 
'''

#from functools import cache


class Node:

    def __init__(self, key, value):
         self.key = key
         self.val = value
         self.next = self.prev = None
         
class LRUCache:

    def __init__(self, size):
        self.cap = size
        self.cache = {}

        #doubly linked list
        # LRU
        self.left = Node(0,0)
        # MRU
        self.right = Node(0,0)

        self.left.next, self.right.prev = self.right, self.left
    
    def remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev
    
    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.next = nxt.prev = node
        node.next, node.prev = nxt, prev

    def get(self, key):
        if key in self.cache:
            # return the val and update the MRU
            self.remove(self.cache[key]) # remove this and make it MRU
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

    def put(self, key, value):
        if key in self.cache:
            #remove the key and insert the updated one
            self.remove(self.cache[key])
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])

        if len(self.cache) > self.cap:
            #remove the LRU
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
    

inputi = ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
val = [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

lenInput = len(inputi)
res = []
cacheSize = val[0][0]
lrucache = LRUCache(cacheSize)

for i in range(1, lenInput):
    
    if inputi[i] == "put":
        key = val[i][0]
        value = val[i][1]
        lrucache.put(key,value)
        res.append("null")
    
    elif inputi[i] == "get":
        key = val[i][0]
        value = lrucache.get(key)
        res.append(value)

print("printing result",res)