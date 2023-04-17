import sys
import os
import numpy as np

### important! 중요!
# If the current working directory is ch4, please work from the parent directory.
parent_path = os.getcwd()
sys.path.append(parent_path) 

from dataset.mnist import load_mnist
from gradient_twolayernet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)


# layer size
input_size = 784 # 26x26
hidden_size = 50
output_size = 10 # 10 class
network = TwoLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# hyper parameters
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
iter_nums = 10000

# train, test (SGD)
train_loss_list = []
train_acc_list = []
test_acc_list = []

import math
iter_per_epoch = max( math.ceil(train_size / batch_size), 1)

for i in range(iter_nums):
    # batch_mask (SGD)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # gradient
    gradient = network.gradient(x_batch, t_batch)
    
    # parameter update
    for key in ['W1', 'b1', 'W2', 'b2']: 
        network.params[key] = network.params[key] - learning_rate*gradient[key]
    