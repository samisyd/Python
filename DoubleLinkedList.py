class Node:

    def __init__(self, data):
        self.data = data
        self.next = None
        self.previous = None

class DoublyLinkedList:

    def __init__(self):
         self.head = None

    def append(self, data):
        new_node = Node(data)
        curr = self.head

        if curr is None:
            self.head = new_node
            return
        else:
            while curr.next != None:
                curr = curr.next
            
            curr.next = new_node
            new_node.previous = curr

    def size(self):

        count = 1
        curr = self.head
        if self.head is None:
            return 0
        else:
            while curr.next:
                count += 1
                curr = curr.next
        return count

    def print_ddl(self):
        
        curr = self.head
        if self.head is None:
            print("empty ddl")
        else:
            while curr.next:                
                print(str(curr.data))
                curr = curr.next
            print(str(curr.data))

    
    def remove_node(self, data):

        curr = self.head

        if curr is None:
            print("list is empty")
            return
        else:
            if self.head.data == data:
                n = self.head
                if self.head.next:
                    self.head = self.head.next
                    self.head.previous = None
                n = None
                return

            while curr:
                if curr.data != data:
                   curr = curr.next
                else:                              
                    node = curr
                    prev = curr.previous
                    prev.next = curr.next
                    if curr.next :
                        curr.next.previous = prev
                    node = None
                    return
        print("data not found")
        

ddl = DoublyLinkedList()
ddl.append(3)
ddl.append(7)
ddl.append(9)
ddl.print_ddl()
ddl.remove_node(7)
print("\n")
ddl.print_ddl()