import sys
import os
import numpy as np

### important! 중요!
# If the current working directory is ch4, please work from the parent directory.
parent_path = os.getcwd()
sys.path.append(parent_path) 

from common.activation_functions import softmax, sigmoid, sigmoid_grad
from common.loss_functions import cross_entropy_error
from common.gradient import numerical_gradient

from collections import OrderedDict
from common.layers import *

# *E* :: extend

class MultiLayerNetExtend:
    """
    # Fully Connected Multi Layer Network
    # weight_decay, dropout, batch_norm

    * input_size: network input size
    * hidden_size_list: each hidden layer's size (ex [100,60,100])
    * output_size: network output size (# of categories)

    * activation: activation function (relu or sigmoid)
    * weight_init_std:  'relu' or 'he' -> 'He' (2/sqrt(n))
                        'sigmoid' or 'xavier' -> 'Xavier' (1/sqrt(n))

    * weight_decay_lambda: weight decay lambda

    *E* use_dropout : 드롭아웃 사용 여부
    *E* dropout_ration : 드롭아웃 비율

    *E* use_batchNorm : 배치 정규화 사용 여부
    """
    def __init__(self, input_size, hidden_size_list, output_size, 
                 activation='relu', weight_init_std='relu', 
                 weight_decay_lambda = 0.001,
                 use_dropout = False, dropout_ratio = 0.5, use_batchnorm=False):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)

        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self.use_batchnorm = use_batchnorm

        # initialize weights
        self.__init_weight(weight_init_std)

        # *E* initialize layers 
        self.__init_layer(activation, use_dropout, dropout_ratio, use_batchnorm)
    
    def __init_weight(self, weight_init_std):
        """
        # initialize weight
        * weigt_init_std: 'relu' or 'he' -> 'He' (2/sqrt(n))
                        'sigmoid' or 'xavier' -> 'Xavier' (1/sqrt(n))
        """
        weight_init_std = str(weight_init_std).lower()

        network_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(network_size_list)):
            # He
            if weight_init_std in ['relu' , 'he']:
                std_scale = np.sqrt(2.0 / network_size_list[idx-1])
            # Xavier
            elif weight_init_std in ['sigmoid' , 'xavier']:
                std_scale = np.sqrt(1.0 / network_size_list[idx-1])
            # number value
            else:
                std_scale = float(weight_init_std)

            # weight
            self.params['W'+str(idx)] = std_scale * np.random.randn(network_size_list[idx-1], network_size_list[idx])
            # bias
            self.params['b'+str(idx)] =  np.zeros(network_size_list[idx])

    def __init_layer(self, activation, use_dropout, dropout_ratio, use_batchnorm):
        """
        # *E* initialize layer
        * activation: 'relu' or 'sigmoid'

        *E* use_dropout: True/False
        *E* dropout_ratio: (ex: 0.5)

        *E* use_batchnorm: True/False
        """
        activation_layer = {'relu': Relu, 'sigmoid': Sigmoid}
        self.layers = OrderedDict()
        
        # hidden layers
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

            # *E* batchnorm
            if use_batchnorm:
                # gamma, beta
                self.params['gamma' + str(idx)] = np.ones(self.hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(self.hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            self.layers['Activation' + str(idx)] = activation_layer[activation]()

            # *E* dropout
            if use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio)
        
        # last layer
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

 
    def predict(self, x, train_flag=False):
        ### ch5: predict with layers
        # *E* key: Dropout , BatchNorm : only training
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flag=False):
        ### ch5: last_layer
        y = self.predict(x, train_flag)

        # not using weight decay
        if self.weight_decay_lambda <= 0:
            return self.last_layer.forward(y, t)

        # using weight decay
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)

        # get max index
        y = np.argmax(y, axis=1)

        # batch support
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        correct = np.sum(y==t)
        accuracy = correct / y.shape[0]

        return accuracy

    ### ch5 contents, very fast
    def gradient(self, x, t):
        ### ch5: with layers
        
        # *E* forward
        self.loss(x, t, train_flag=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # grads
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            # weight decay or not
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW \
                + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            # *E*: batchnorm -> gamma, beta too.
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads

    # ch4 contents, very slow
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            # *E*: batchnorm -> gamma, beta too.
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])
        
        return grads

