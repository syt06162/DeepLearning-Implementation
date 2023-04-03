import numpy as np

def sum_of_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def mean_squared_error(y, t):
    return np.sum((y-t)**2) / len(y) 

# # (no batch) cross_entropy_error
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))

# (batch, normalized) cross_entropy_error 
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y.reshape(1, y.shape[0])
        t.reshape(1, t.shape[0])
    
    batch_size = y.shape[0]
    delta = 1e-7  
    return -np.sum(t * np.log(y + delta)) / batch_size

