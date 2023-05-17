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

class AdaGrad:
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
            # h(t) <- (D) * h(t-1) + (1-D) * (dL*dL)
            self.h[key] = self.decay_rate * self.h[key] + (1 - self.decay_rate) * (grads[key] * grads[key])
            params[key] -= self.lr*grads[key] / (np.sqrt(self.h[key]) + 1e-7) # avoid zero division

class Adam:
    """ beta1 for momentum, beta2 for adagrad """
    def __init__(self, lr = 0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None

        # iter?
        self.iter = 0
    
    def update(self, params, grads):
        # m, v initialize
        if self.m == None:
            self.m , self.v = {}, {}
            for key, value in params.items():
                self.m[key] = np.zeros_like(value)
                self.v[key] = np.zeros_like(value)

        # iter
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            # m(t) <- (b1) * m(t-1) + (1-b1) * (dL)
            # v(t) <- (b2) * v(t-1) + (1-b2) * (dL*dL)
            self.m[key] = self.beta1 * (self.m[key]) + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * (self.v[key]) + (1 - self.beta2) * grads[key] * grads[key]
            # params
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7) # avoid zero division
