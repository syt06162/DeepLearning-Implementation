import sys
import os
import numpy as np

### important! 중요!
# If the current working directory is ch3, please work from the parent directory.
parent_path = os.getcwd()
sys.path.append(parent_path) 

from common.activation_functions import softmax
from common.loss_functions import cross_entropy_error
from common.gradient import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        z = np.dot(x, self.W)
        return z

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss_val = cross_entropy_error(y, t)
        return loss_val


# test
if __name__ == "__main__":
    net = SimpleNet()
    
    # print W
    print("Weight parameters (random)")
    print(net.W)
    print()
    
    # print predict
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print("predict array")
    print(p)
    print()

    # pritn true, loss
    t = np.array([0, 0, 1])
    print("loss:")
    print(net.loss(x, t))
    print()

    # numerical_gradient 
    def f(W):
        return net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print("numerical_gradient:")
    print(dW)

