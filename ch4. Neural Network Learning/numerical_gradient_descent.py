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

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr*grad
    
    return x


# test
if __name__ == "__main__":

    # numerical gradient
    def fun_1(x):
        return x[0]**2 + x[1]**2
    
    print("numerical_gradient: ")
    print("f(x0, x1) = x0² + x1²")
    print([3.0, 4.0], "->" , numerical_gradient(fun_1, np.array([3.0, 4.0])))  
    print([0.0, 2.0], "->" , numerical_gradient(fun_1, np.array([0.0, 2.0])))
    print()

    # gradient descent
    print("gradient_descent: ")
    print("f(x0, x1) = x0² + x1²")
    start_position = [-3.0, 4.0]
    lr = 0.001
    print(start_position, "lr=" + str(lr), "->" , gradient_descent(fun_1, np.array(start_position), lr=lr))  
    lr = 1
    print(start_position, "lr=" + str(lr), "->" , gradient_descent(fun_1, np.array(start_position), lr=lr))
    print()
