import numpy as np

def step_function(x):
    return np.array(x > 0, dtype = np.int32)

def sigmoid(x):
    return 1 / (np.exp(-x)+1)

def sigmoid_grad(x):
    return (1 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def identity(x):
    return x

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    elif x.ndim == 1:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
