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
        return dout


if __name__ == "__main__":
    x = np.array([-2, -1, 3, 10])
    relu = Relu()
    print(relu.forward(x))
    print(relu.backward(x))

    