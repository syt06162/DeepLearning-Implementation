class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        # x, y -> x*y
        self.x = x
        self.y = y
        out = x*y
        return out
    
    def backward(self, dout):
        # x, y -> dout*y, dout*x
        dx = dout*self.y
        dy = dout*self.x
        return dx, dy


class AddLayer:
    def forward(self, x, y):
        # x, y -> x+y
        out = x+y
        return out
    
    def backward(self, dout):
        # x, y -> dout, dout
        dx = dout
        dy = dout
        return dx, dy
