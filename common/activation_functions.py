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
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x