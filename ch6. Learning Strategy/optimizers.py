import numpy as np

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr = 0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        # v initialize
        if self.v == None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)

        for key in params.keys():
            self.v[key] = self.v[key] * self.momentum - self.lr * grads[key]
            params[key] += self.v[key]

class Adagrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        # h initialize
        if self.h == None:
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr*grads[key] / (np.sqrt(self.h[key]) + 1e-7) # avoid zero division

class RMSprop:
    def __init__(self, lr = 0.01, decay_rate=0.99):
        self.lr = lr
        self.h = None
        self.decay_rate = decay_rate
    
    def update(self, params, grads):
        # h initialize
        if self.h == None:
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)

        for key in params.keys():
            # h(t) <- (D) * h(t-1) + (1-D)(dL*dL)
            self.h[key] = self.decay_rate * self.h[key] + (1 - self.decay_rate) * (grads[key] * grads[key])
            params[key] -= self.lr*grads[key] / (np.sqrt(self.h[key]) + 1e-7) # avoid zero division
