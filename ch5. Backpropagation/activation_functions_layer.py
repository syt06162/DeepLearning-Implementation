import numpy as np

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (np.exp(-x)+1)
        return self.out
    
    def backward(self, dout):
        dx = dout * (self.out) * (1 - self.out)
        return dx


if __name__ == "__main__":
   
    # relu test
    test = [-2, -1, 3, 10]
    x = np.array(test)
    relu = Relu()

    print("# relu test")
    print(test)
    print("forward:")
    print(relu.forward(x))
    print("backward:")
    print(relu.backward(x))
    print()


    # sigmoid test
    test = [-2, -1, 3, 10]
    x = np.array(test)
    sigmoid = Sigmoid()

    print("# sigmoid test")
    print(test)
    print("forward:")
    print(sigmoid.forward(x))
    print("backward:")
    print(sigmoid.backward(x))
    print()