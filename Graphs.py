# implementation of an undirected graph using Adjacency Lists
'''
adj = {
    v1 : [v2,v3],
    v2 : [v1,v4]
}
'''
class Vertex:
	def __init__(self, n):
		self.name = n
		self.neighbors = list()
	
	def add_neighbor(self, v, weight):
		if v not in self.neighbors:
			self.neighbors.append((v, weight))
			self.neighbors.sort()

class Graph:
	vertices = {}
	
	def add_vertex(self, vertex):
		if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
			self.vertices[vertex.name] = vertex
			return True
		else:
			return False
	
	def add_edge(self, u, v, weight=0):
		if u in self.vertices and v in self.vertices:
			
			self.vertices[u].add_neighbor(v, weight)
			self.vertices[v].add_neighbor(u, weight)
			return True
		else:
			return False
			
	def print_graph(self):
		for key in sorted(list(self.vertices.keys())):
			print(key + str(self.vertices[key].neighbors))

g = Graph()
# print(str(len(g.vertices)))
a = Vertex('A')
g.add_vertex(a)
g.add_vertex(Vertex('B'))
# ORD A - 65 == k = 75
for i in range(ord('A'), ord('K')):
	g.add_vertex(Vertex(chr(i)))

edges = ['AB', 'AE', 'BF', 'CG', 'DE', 'DH', 'EH', 'FG', 'FI', 'FJ', 'GJ', 'HI']
for edge in edges:
	g.add_edge(edge[:1], edge[1:])

g.print_graph()


#slidw(Arr, 4)
#Given a directed graph. The task is to do Breadth First Traversal of this graph starting from 0. 
# You don’t need to read input or print anything. Your task is to complete the function bfsOfGraph()
#  which takes the integer V denoting the number of vertices and adjacency list as input parameters and 
# returns  a list containing the BFS traversal of the graph starting from the 0th vertex from left to right.
# input - V=5 adj = [[0,1], [0,2], [0,3], [2,4]] Output - 0 1 2 3 4
def bfsOfGraph(V, adj):
        if len(adj) == 0 or V == 0:
            return []
        
        # code here
        graph = [[] for i in range(V)]
        #for i in range(len(adj)):
         #   graph[adj[i][0]].append(adj[i][1])
          #  graph[adj[i][1]].append(adj[i][0])
        #print(graph)

        for x,y in adj:
            graph[x].append(y)
            graph[y].append(x)
        print(graph)

        result = []
        queue = []
        visited = set()
        queue.append(0)
        visited.add(0)
        i = 0
        while i < len(queue):
            popped = queue[i]
            i+=1
            result.append(popped)
            
            for vertex in graph[popped]:
                if vertex not in visited:
                    queue.append(vertex)
                    visited.add(vertex)
            
        return result


def dfsOfGraph(V, adj):
        # code here

        graph = [[] for i in range(V)]

        for x,y in adj:
            graph[x].append(y)
            graph[y].append(x)
        print(graph)

        result = []
        visited = set()
        visited.add(0)
        
        def dfsHelper(v):
            result.append(v)
            for vertex in graph[v]:
                if vertex not in visited:
                    visited.add(vertex)
                    dfsHelper(vertex)        
                    
        dfsHelper(0)
        return result

V=5 
adj = [[0,1], [0,2], [0,3], [2,4]]
print("printing bfs", bfsOfGraph(V, adj))
print("printing dfs", dfsOfGraph(V, adj))


'''
https://www.hackerrank.com/challenges/bfsshortreach/problem
Consider an undirected graph where each edge weighs 6 units. Each of the nodes is
 labeled consecutively from 1 to n.

You will be given a number of queries. For each query, you will be given a list of 
edges describing an undirected graph. After you create a representation of the graph, 

you must determine and report the shortest distance to each of the other nodes from a given 
starting position using the breadth-first search algorithm (BFS). Return an array of distances 
from the start node in node number order. If a node is unreachable, return -1
for that node. 
Sample Input

2   - no of queries

4 2 - 4= nodes 2 - edges
1 2 - u,v - describe an edge
1 3 - u,v - describe an edge
1   - starting node

3 1 - noofNodes, noOfedges
2 3 - u,v
2   - starting node

Sample Output

6 6 -1
-1 6

'''
import collections
import math
import os
import random
import re
import sys
from collections import deque

#
# Complete the 'bfs' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER m
#  3. 2D_INTEGER_ARRAY edges
#  4. INTEGER s
#
'''
n = 5 # no of vertices
m = 3 # no of edges
edges = [[1,2], [1,3], [3,4]]
s = 1 #starting node
'''

def bfs(n, m, edges, s):
    # Write your code here
    graph = [[] for i in range(n+1)]
    
    for x,y in edges:
        graph[x].append(y)
        graph[y].append(x)
    print(graph)
       
    visited = set()
    distances = [-1] * (n+1)
    q = deque([(s, 0)])
    distances[s] = 0
    visited.add(s)
    
    while len(q):
        v, w = q.popleft()       
        for vertex in graph[v]:
            if vertex not in visited:                
                distances[vertex] = w+6
                q.append((vertex, w+6))
                visited.add(vertex)                
    
    distances.remove(0) #remove the 0th element as not part of the vertices
    print(distances)
    
    return distances[1:]

def dfs(n, m, edges, s):
    # Write your code here
    graph = [[] for i in range(n+1)]
    
    for x,y in edges:
        graph[x].append(y)
        graph[y].append(x)

    stack = []  
    visited = set()
    distances = [-1] * (n+1)
    stack.append([s,0])
    distances[s] = 0
    visited.add(s)
    
    while len(stack):
        v, w = stack.pop()       
        for vertex in graph[v]:
            if vertex not in visited:                
                distances[vertex] = w+6
                stack.append([vertex, w+6])
                visited.add(vertex)    
    
    distances.remove(0) # remove the index with value 0  - basically the starting node
    return distances[1:]

n = 5 # no of vertices
m = 3 # no of edges
edges = [[1,2], [1,3], [3,4]]

s = 1 #starting node
print("printing weighted graph", bfs(n, m, edges, s))

print("printing weighted dfs graph",dfs(n, m, edges, s))




#n = 4 # no of vertices
#m = 2 # no of edges
#edges = [[1,2], [1,3]]
#s = 1

n = 3 # no of vertices
m = 1 # no of edges
edges = [[2,3]]
s = 2 #starting node
print("printing weighted graph ==>2 ", bfs(n, m, edges, s))

print("printing weighted dfs graph == > 2",dfs(n, m, edges, s))


#class Solution:
    #Function to return the adjacency list for each vertex.
def printGraph(V, adj):
    # code here
    results = []

    graph = [[] for i in range(V+1)]
    for x,y in adj:
        graph[x].append(y)
        graph[y].append(x)
    print(graph)

    
    for i in range(V+1):
        result =[]    
        result.append(i)
        print(result)
        for j in graph[i]:
            result.append(j)
        print("in  j",result)
        results.append(result)
    return results

V = 4 # no of vertices
#m = 3 # no of edges
adj = [[1,2], [1,3], [3,4]]

print(printGraph(V, adj))

'''
# give 1 first, then give 5 7 -(represnts V E)
# then give values to store in list of lists 0 1, 0 4, 1 2, 1 3, 1 4, 2 3, 3 4 
T = int(input())
for i in range(T):

    V, E = map(int, input().split())
    print(V, E)
    adj = [[] for i in range(V)]
    print(adj)
    for _ in range(E):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    obj = Solution()
    ans = obj.printGraph(V, adj)
    print(ans)
    for i in range(len(ans)):
        for j in range(len(ans[i]) -1):
            print(ans[i][j], end = "->")
        print(ans[i][len(ans[i])-1])
'''


'''
https://leetcode.com/problems/min-cost-to-connect-all-points/

Prims algo - leet code 1584
You are given an array points representing integer coordinates of some points 
on a 2D-plane, where points[i] = [xi, yi]. The cost of connecting two points [xi, yi] 
and [xj, yj] is the manhattan distance between them: |xi - xj| + |yi - yj|, 
where |val| denotes the absolute value of val. Return the minimum cost to make all
points connected. All points are connected if there is exactly one simple
path between any two points.

Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Output: 20
Explanation: 

We can connect the points as shown above to get the minimum cost of 20.
Notice that there is a unique path between every pair of points.

Input: points = [[3,12],[-2,5],[-4,1]]
Output: 18

Constraints:

    1 <= points.length <= 1000
    -106 <= xi, yi <= 106
    All pairs (xi, yi) are distinct.

this ssolution takes n^2 log n time
n^2 for inserting each connected node of arr n times to heap
and logn to pop an element from heap
'''
import heapq
# these are x,y cordinates
points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
def primsAlgo(points):

    # Build the adjacency list first
    n = len(points)
    adj = {i:[] for i in range(n)} # i : list of [cost, node]

    for i in range(n):
        x,y = points[i]
        for j in range(i+1, n):
            x2,y2 = points[j]
            dis = abs(x-x2) + abs(y-y2)
            adj[i].append([dis, j])
            adj[j].append([dis, i])

    print("printing adj",adj)

    # Prins algo using min heap
    minheap = [[0,0]]
    visited = set()
    costs = 0

    while len(visited) < n:

        dis,node = heapq.heappop(minheap)        
        if node not in visited:
            visited.add(node)
            costs += dis
        else:
            continue
        for neidis, nei in adj[node]:
            if nei not in visited:
                heapq.heappush(minheap, [neidis, nei])
    
    return costs

points2 = [[3,12],[-2,5],[-4,1]]
print("Cost of spanning tree ==>", primsAlgo(points))

#https://www.hackerrank.com/challenges/primsmstsub/
def prims(n, edges, start):
    # Write your code here
    
    adj = {i:[] for i in range(1,n+1)}
    m = len(edges)
    for u,v,w in edges:
        adj[u].append([v,w])
        adj[v].append([u,w])

    print(adj)

    minhp = [[0, start]] # add [wt, node] to minHeap
    totalcost = 0
    visited = set()
    while len(visited) < n:

        weight, node = heapq.heappop(minhp)
        
        if node in visited:
            continue
        
        visited.add(node)
        totalcost += weight
        for nei, wghtnei in adj[node]:
            if nei not in visited:
                heapq.heappush(minhp, [wghtnei, nei])

    return totalcost


n1 = 3
edges = [[1,2,2], [2,3,2], [1,3,3]]
n2 = 5
edges2 = [[1,2,3], [1,3,4], [4,2,6], [5,2,2], [2,3,5], [3,5,7]]
print("prims =>",prims(n2, edges2, 1))


'''
The member states of the UN are planning to send people to the moon. 
They want them to be from different countries. You will be given a 
list of pairs of astronaut ID's. Each pair is made of astronauts 
from the same country. Determine how many pairs of astronauts
from different countries they can choose from.
https://www.youtube.com/watch?v=IeZs94EFCTk
https://www.youtube.com/watch?v=956gDtlo_jQ
https://www.youtube.com/watch?v=dP1Auzs1yJo

'''
n = 4
astronaut1 = [[0,1], [2,3], [0,4]]
astronaut = [[1,2], [2,3]]
def journeyToMoon(n, astronaut):
    # Write your code here
    
    graph = {i:[] for i in range(n)}
    for x,y in astronaut:
        graph[x].append(y)
        graph[y].append(x)

    print(graph)

    results = []
    visited = set()

    def dfs(node, bfsRes):

        visited.add(node)
        bfsRes.append(node)
        
        for neigh in graph[node]:            
            if neigh not in visited:
                dfs(neigh, bfsRes)

    #do DFS and find the pair of astronauts from same country
    for i in range(n):
        bfsRes = []
        if i not in visited:
            dfs(i, bfsRes)        
            results.append(bfsRes)
    print(results)

    # newanswer = oldanswer + sumofoldvalues * newvalue
    sumofoldvalues = len(results[0])
    ans =0 
    
    for i in range(1, len(results)):
        newval = len(results[i])        
        ans = ans + sumofoldvalues * newval
        sumofoldvalues = sumofoldvalues +  newval

    print("answer is ",ans)

journeyToMoon(n, astronaut)

'''
input - V=5 adj = [[0,1], [0,2], [0,3], [2,4]] Output - 0 1 2 3 4
Given a directed graph. The task is to do Breadth First Traversal of this graph starting from 0. 
'''
V = 5
edjes = [[0,1], [0,2], [0,3], [2,4]]

def mybfsGraph(V, edjes):

    adjL = {i:[] for i in range(V)}

    for u,v in edjes:
        adjL[u].append(v)
        adjL[v].append(u)

    print("my list ",adjL)
    res = []
    visited = set()
    start = 0
    q = collections.deque()
    q.append(0)
    visited.add(0)
    while q:

        vertex = q.popleft()
        res.append(vertex)
        for nei in adjL[vertex]:
            if nei not in visited:
                visited.add(nei)
                q.append(nei)

    print(res)
    
mybfsGraph(V, edjes)
'''
Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Output: 20
Explanation: 

We can connect the points as shown above to get the minimum cost of 20.
Notice that there is a unique path between every pair of points.

Input: points = [[3,12],[-2,5],[-4,1]]
Output: 18

Constraints:

    1 <= points.length <= 1000
    -106 <= xi, yi <= 106
    All pairs (xi, yi) are distinct.

this ssolution takes n^2 log n time
n^2 for inserting each connected node of arr n times to heap
and logn to pop an element from heap
'''

import heapq
# these are x,y cordinates
points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
def primsAlgo(points):

    N = len(points)
    # will create {0:[], 1:[], 2:[]....} 5 points on the graph x,y
    # and weights are |x1-x2| + |y1-y2| 
    adjl = {i:[] for i in range(N)}
    for i in range(N):
        x1,y1 = points[i]
        for j in range(i+1, N):
            x2,y2 = points[j]
            #calc manhattan distance
            distance = abs(x1-x2) + abs(y1-y2)
            adjl[i].append([distance,j])
            adjl[j].append([distance,i])

    print(adjl)
    '''
    list contains distance for all nodes : ex distance from node0 to node1
    created adj list is node 0 : [distance1, node1],[distance2, node2]
    {0: [[4, 1], [13, 2], [7, 3], [7, 4]], 1: [[4, 0], [9, 2], [3, 3], [7, 4]],
     2: [[13, 0], [9, 1], [10, 3], [14, 4]], 3: [[7, 0], [3, 1], [10, 2], [4, 4]], 4: [[7, 0], [7, 1], [14, 2], [4, 3]]}
    '''
    # create min heap starting cost is 0 with starting node 0
    minh = [[0,0]]
    visited = set()
    rescost = 0
    while len(visited) < N:

        costDis,node = heapq.heappop(minh)
        if node in visited:
            continue
        visited.add(node)
        rescost += costDis
        for costnei, nei in adjl[node]:
            heapq.heappush(minh, [costnei, nei])

    print("cost of min spanningtree is ",rescost)

primsAlgo(points)

'''
https://www.youtube.com/watch?v=EaphyqKU4PQ
Medium
Network Delay Time - Dijkstra's algorithm - Leetcode 743
You are given a network of n nodes, labeled from 1 to n. You are also given times,
 a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is 
 the source node, vi is the target node, and wi is the time it takes for a 
 signal to travel from source to target.

We will send a signal from a given node k. Return the minimum time it takes for
 all the n nodes to receive the signal. If it is impossible for all the n nodes
to receive the signal, return -1.

Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2

    edges = dict()
    for u,v,w in times:
        if u in edges:
            edges[u].append([v,w])
        else:
            edges[u] = [[v,w]]

What is the difference between Prim and Dijkstra?

Comparing Prims and Dijkstra Algorithms

we have the MSpanning tree T of the graph . 
Note that an MST always contains all vertices of the original graph.

Shortest path -
a shortest-path tree may not contain all vertices of the original graph.

In the computation aspect, Prims and Dijkstras algorithms have three main differences:

    Dijkstras algorithm finds the shortest path, but Prims algorithm finds the MST
    Dijkstras algorithm can work on both directed and undirected graphs, 
    but Prims algorithm only works on undirected graphs
    Prims algorithm can handle negative edge weights, but Dijkstras algorithm 
    may fail to accurately compute distances if at least one negative edge weight exists

In practice, Dijkstras algorithm is used when we want to save time and fuel 
traveling from one point to another. Prims algorithm, on the other hand, is used 
when we want to minimize material costs in constructing roads that connect
multiple points to each other.
   

https://www.baeldung.com/cs/prim-dijkstra-difference
'''
# Dijkstra shortest path algorithm
# this is a directional graph 
times = [[2,1,1],[2,3,1],[3,4,1]]
times2 = [[1,2,5],[1,3,1],[3,2,1],[3,4,2],[4,2,1]]
n = 4
k = 1
def NetworkDelay(times, n, k):

    # the defaultdict does not throw an error if the key is not already present.
    # It provides a default value for the key that does not exists.
    
    #edges = collections.defaultdict(list)
    #for u,v,w in times:
    #   edges[u].append([w,v])

    edges = dict()
    vertex = set()
    for u,v,w in times:
        vertex.add(u)
        vertex.add(v)
        if u in edges:
            edges[u].append([w,v])
        else:
            edges[u] = [[w,v]]
    print(edges)
    print(vertex)
    # add missing vertex
    for item in vertex:
        if item not in edges:
            edges[item] = []

    minh = [[0, k]]
    visited = set()
    totalCost = 0
    while minh and len(visited) < n:
        cost, node = heapq.heappop(minh)
        if node in visited:
            continue
        visited.add(node)
        print(visited)
        totalCost = max(totalCost, cost) 
        for neiCost, nei in edges[node]:
            if nei not in visited:
                heapq.heappush(minh, (neiCost + cost, nei))
    
    return totalCost if len(visited) == n else -1


print(NetworkDelay(times2, n, k))

'''
stri = '+1.0'
stri = '-1.0'
stri = '1.4.22'
stri = '40.0'
stri = '40c.55'
stri = 'a'
'''
def strparse(stri):

    countDot = 0
    for i in range(len(stri)):

        if stri[i] == '.':
            countDot +=1
            if countDot > 1:
                return False
        if stri[i].isalpha():
            return False
        if i >=1 :
            if stri[i] == '+'  or stri[i] == '-':
                return False
        
    return True

print(strparse('-1.0'))
print(strparse('+1.0'))
print(strparse('1.4.22.0'))
print(strparse('40.0'))
print(strparse('a'))
print(strparse('10+0'))