
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
    
test_array = [(0,0), (0,1), (1,0), (1,1)]
for x1, x2 in test_array:
    print(AND(x1,x2))