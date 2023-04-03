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


# test
if __name__ == "__main__":

    y = [0.10, 0.05, 0.00, 0.60, 0.05, 0.10, 0.00, 0.00, 0.10, 0.0]
    t = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    # SSE
    print("SSE:", sum_of_squared_error(np.array(y), np.array(t)))

    # MSE
    print("MSE:", mean_squared_error(np.array(y), np.array(t)))

    # CEE
    print("CEE:", cross_entropy_error(np.array(y), np.array(t)))