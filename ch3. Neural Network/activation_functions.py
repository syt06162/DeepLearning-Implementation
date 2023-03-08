import numpy as np

def step_function(x):
    return np.array(x > 0, dtype = np.int32)

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