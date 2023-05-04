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
    
    # train/test loss and accuracy (per epoch)
    if i % iter_per_epoch == 0:
        train_loss = network.loss(x_train, t_train)
        train_loss_list.append(train_loss)

        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print("epoch {:3d} :: train loss | {:.3f}".format(int((i/iter_per_epoch)), train_loss))
        print("train acc, test acc | {:.3f}, {:.3f}".format(train_acc, test_acc))
        print()
        

# final result
print("final result :: epoch {:3d}".format(int(iter_nums/iter_per_epoch)))
print("train acc, test acc | {:.3f}, {:.3f}".format(train_acc, test_acc))


# plot
import matplotlib.pyplot as plt 
epochs = [i for i in range(len(train_acc_list))]

# Plotting train and test accuracy
plt.plot(epochs, train_acc_list, label='Train Accuracy')
plt.plot(epochs, test_acc_list, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting train loss
plt.figure()
plt.plot(epochs, train_loss_list, label='Train Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()