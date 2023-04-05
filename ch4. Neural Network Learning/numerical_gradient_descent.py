import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    for i in range(np.size(x)):
        x_i = x[i]

        # f(x+h)
        x[i] = x_i + h
        fxh1 = f(x)

        # f(x-h)
        x[i] = x_i - h
        fxh2 = f(x)

        gradient[i] = (fxh1 - fxh2) / (2*h)
        x[i] = x_i # recover
        
    return gradient


# test
if __name__ == "__main__":

    def fun_1(x):
        return x[0]**2 + x[1]**2
    
    print(numerical_gradient(fun_1, np.array([3.0, 4.0])))  
    print(numerical_gradient(fun_1, np.array([0.0, 2.0])))
    
