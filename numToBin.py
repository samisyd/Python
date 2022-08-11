


def get_bin(num):
    
    bin = []
    while num > 0 :
        remainder = num % 2
        num = num // 2
        bin.append(remainder)

    print(bin)
    bin = bin[::-1]
    str1 = "".join(list(map(str, bin)))
    
    
    print("enter2")
    print(str1)

print("enter")
get_bin(242)


def is_match(value1, value2):

    if value1 == '[' and value2 == ']':
        return True

    elif value1 == '{' and value2 == '}':
        return True
        
    elif value1 == '(' and value2 == ')':
        return True
    
    else:
        return False

def is_balanced_param(instr):

    paramstack = []

    if len(instr) == 0 :
        return True
    for elem in instr:

        if elem in '([{':
            paramstack.append(elem)
        else:
            if not len(paramstack):
                return False
            else :
                val = paramstack.pop()
                if not is_match(val, elem):
                    return False

    if not len(paramstack):
        return True

print(is_balanced_param("()"))
print(is_balanced_param("())"))
print(is_balanced_param("[{()}]"))
print(is_balanced_param(""))
print(is_balanced_param(")"))

'''
def get_rev(str1):

    s = Stack()
    for ele in str1:
        s.push(ele)

    rev_str = ""
    while not s.is_empty():
        rev_str += s.pop()

    
    return rev_str
print(get_rev("samina"))
'''

# stock prices 

def solution(prices):
    # Type your solution here
    n = len(prices)
    if n == 0:
        return 0
    
    l = 0
    r = 1
    maxprofit = float('-inf')
    print(maxprofit)
    while l<r and r <n:
        if prices[r]< prices[l]:
            l = r            
        else:            
            val = prices[r] - prices[l] 
            maxprofit = max(maxprofit, val)
        
        r+=1
    
    return maxprofit

prices = [6, 0 , -1, 10]
print(solution(prices))

res = ['<' for  i in range(3)]
print(res)

# Given a string , for the result with proper string match in o/p
stri = '><<>>><<'
stri2 = '<<>>><><<'
def braces(stri):
    
    stackop = []
    stackcl = []
    res = []
    i = 1
    m = len(stri)
    
    stackcl.append(stri[0]) if stri[0] == '>' else stackop.append(stri[0])
    
    while stackcl or stackop and i<m:

        if stri[i] == '<':
            if stackcl :
                size = len(stackcl)
                while stackcl:      
                    stackcl.pop()
                    res.append('<')
                while size :
                    res.append('>')
                    size -=1
            stackop.append('<')
            i+=1
        elif stri[i] == '>':
            if stackop :
                stackop.pop()
            stackcl.append('>')
            i +=1

    if stackcl:
        size = len(stackcl)
        while stackcl:      
            stackcl.pop()
            res.append('<')
        while size :
            res.append('>')
            size -=1
    if stackop:
        size = len(stackop)
        while stackop:      
            stackop.pop()
            res.append('<')
        while size :
            res.append('>')
            size -=1
    
    res = "".join(res)
    return res

print("i/p is value",stri2)
print("o/p is")
print(braces(stri2))
