import numpy as np

def step_function(x):
    return np.array(x > 0, dtype = np.int32)

def sigmoid(x):
    return 1 / (np.exp(-x)+1)

def relu(x):
    return np.maximum(0, x)

# step test
a = np.array([1, -1])
print(a)
b = (a>0).astype(np.int32)
print(b)

print(step_function(a))
print(step_function(1))
print(step_function(2))
print(step_function(0))
print(step_function(-1))

# sigmoid test
print(sigmoid(1))
print(sigmoid(np.array([0,2, -2])))

# relu test
print(relu(0))
print(relu(np.array([-3,-1,1,4])))