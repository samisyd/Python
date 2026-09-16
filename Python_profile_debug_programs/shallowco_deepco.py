import copy

original = [
    ["Alice", 25],
    ["Bob", 30]
]

shallow = copy.copy(original)

# Change the nested list
shallow[0][1] = 99

print("Original:", original)
print("Shallow: ", shallow)

# ====================================================
original = [
    ["Alice", 25],
    ["Bob", 30]
]
# this 
deep = copy.deepcopy(original)

# Change the nested list
deep[0][1] = 99

print("Original:", original)
print("Deep:    ", deep)