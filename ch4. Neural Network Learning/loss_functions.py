import numpy as np

def sum_of_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def mean_squared_error(y, t):
    return np.sum((y-t)**2) / len(y) 

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))