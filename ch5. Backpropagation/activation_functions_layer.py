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
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
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


    # affine test - batch
    x = np.array( [[1,2], [10,20] ]) # batch
    W = np.array( [[3,4,5] , [6,7,8]] )
    b = np.array( [10,11,12] )
    affine = Affine(W,b)

    print("# affine test")
    print(x)
    print("forward:")
    y = affine.forward(x)
    print(y)
    print("backward:")
    print(affine.backward(y))
    print()

    