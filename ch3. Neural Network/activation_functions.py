import numpy as np

def step_function(x):
    return np.array(x > 0, dtype = np.int32)

def sigmoid(x):
    return 1 / (np.exp(-x)+1)

def relu(x):
    return np.maximum(0, x)

def identity(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


if __name__=="__main__":
    def activation_test(function_name, value):
        print(function_name.__name__ + "(" + str(value) + ") : ", end="")
        print(function_name(value))

    activation_test(step_function, -1)
    activation_test(step_function, np.array([1]))
    activation_test(step_function, np.array([-2,0,4]))
    print()

    activation_test(sigmoid, -1)
    activation_test(sigmoid, np.array([0]))
    activation_test(sigmoid, np.array([-2,0,1,2]))
    print()

    activation_test(relu, -1)
    activation_test(relu, np.array([1]))
    activation_test(relu, np.array([-1,0,1]))
    print()

    activation_test(identity, -1)
    activation_test(identity, np.array([0]))
    activation_test(identity, np.array([-1,0,1]))
    print()

    activation_test(softmax, -1)
    activation_test(softmax, np.array([-1, 1]))
    activation_test(softmax, np.array([50,53,55]))
    print()
