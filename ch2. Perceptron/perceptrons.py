
# Perceptrons

# AND, OR, NOT
# NAND, NOR
# XOR, XNOR

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

def NOT(x1):
    if x1 not in [0,1]:
        return None
    
    w1, b = -1, 0.5

    tmp = np.sum(x1*w1) + b
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

def XOR(x1, x2):
    tmp1 = NAND(x1, x2)
    tmp2 = OR(x1, x2)
    tmp3 = AND(tmp1, tmp2)
    return tmp3

# test
test_array = [(0,0), (0,1), (1,0), (1,1)]
test_array_NOT = [0, 1]
gate_names = [AND, OR, NOT, NAND, XOR]

if __name__ == "__main__":
    for gate_name in gate_names:
        
        print(gate_name.__name__ + " gate:")

        # gate test
        if gate_name.__name__ != "NOT":
            for x1, x2 in test_array:
                print("(", x1 , ",", x2, ") : " , gate_name(x1, x2))
        
        # NOT gate
        else:
            for x1 in test_array_NOT:
                print("(", x1 , ") : " , gate_name(x1))
        print()

