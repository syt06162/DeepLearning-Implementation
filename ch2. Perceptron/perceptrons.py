
# Perceptrons
# AND, OR, NOT, NAND, XOR

import numpy as np

def AND(x1, x2):
    if x1 not in [0,1] or x2 not in [0,1]:
        return None
    
    w1, w2, b = 1, 1, -1.5

    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(x*w) + b
    if tmp > 0:
        return 1
    else:
        return 0

def OR(x1, x2):
    if x1 not in [0,1] or x2 not in [0,1]:
        return None
    
    w1, w2, b = 1, 1, -0.5

    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(x*w) + b
    if tmp > 0:
        return 1
    else:
        return 0

def NOT(x1, x2):
    if x1 not in [0,1] or x2 not in [0,1]:
        return None
    
    w1, w2, b = 0, 1, -0.5

    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(x*w) + b
    if tmp > 0:
        return 1
    else:
        return 0
    
def NAND(x1, x2):
    if x1 not in [0,1] or x2 not in [0,1]:
        return None
    
    w1, w2, b = -1, -1, 1.5

    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(x*w) + b
    if tmp > 0:
        return 1
    else:
        return 0

test_array = [(0,0), (0,1), (1,0), (1,1)]

print("AND")
for x1, x2 in test_array:
    print(AND(x1,x2))

print("OR")
for x1, x2 in test_array:
    print(OR(x1,x2))

print("NOT")
for x1, x2 in test_array:
    print(NOT(x1,x2))

print("NAND")
for x1, x2 in test_array:
    print(NAND(x1,x2))

