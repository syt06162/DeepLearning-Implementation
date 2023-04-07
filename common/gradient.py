import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    # multi-dim support
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        i = it.multi_index # index
        x_i = x[i]

        # f(x+h)
        x[i] = x_i + h
        fxh1 = f(x)

        # f(x-h)
        x[i] = x_i - h
        fxh2 = f(x)

        gradient[i] = (fxh1 - fxh2) / (2*h)
        x[i] = x_i # recover

        it.iternext() # next index
        
    return gradient