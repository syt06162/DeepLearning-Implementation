import numpy as np
import os, sys

### important! 중요!
# If the current working directory is ch4, please work from the parent directory.
parent_path = os.getcwd()
sys.path.append(parent_path) 

from common.activation_functions import softmax, sigmoid
from common.loss_functions import cross_entropy_error

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
        self.out = sigmoid(x)
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
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1): # don't use
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx


if __name__ == "__main__":
   
    # relu test
    test = [-2, -1, 3, 10]
    x = np.array(test)
    relu_test = Relu()

    print("# relu test")
    print(test)
    print("forward:")
    print(relu_test.forward(x))
    print("backward:")
    print(relu_test.backward(x))
    print()


    # sigmoid test
    test = [-2, -1, 3, 10]
    x = np.array(test)
    sigmoid_test = Sigmoid()

    print("# sigmoid test")
    print(test)
    print("forward:")
    print(sigmoid_test.forward(x))
    print("backward:")
    print(sigmoid_test.backward(x))
    print()


    # affine test - batch
    x = np.array( [[1,2], [10,20] ]) # batch
    W = np.array( [[3,4,5] , [6,7,8]] )
    b = np.array( [10,11,12] )
    affine_test = Affine(W,b)

    print("# affine test")
    print(x)
    print("forward:")
    y = affine_test.forward(x)
    print(y)
    print("backward:")
    print(affine_test.backward(y))
    print()


    # softmax-with-loss test
    test = [-2, -1, 3, 4]
    true = [0, 0, 0, 1]
    x = np.array(test)
    t = np.array(true)
    softmax_with_loss_test = SoftmaxWithLoss()

    print("# softmax_with_loss test")
    print(test)
    print("forward:")
    print(softmax_with_loss_test.forward(x, t))
    print("backward:")
    print(softmax_with_loss_test.backward(x))
    print()
