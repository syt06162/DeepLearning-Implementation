import sys
import os
import pickle
import numpy as np

### important! 중요!
# If the current working directory is ch3, please work from the parent directory.
parent_path = os.getcwd()
sys.path.append(parent_path) 

from dataset.mnist import load_mnist
from common.activation_functions import sigmoid, softmax


def load_dataset():
    # mnist dataset
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    
    train_num, image_size = x_train.shape
    test_num = x_test.shape[0]

    # print (x_train, t_train), (x_test, t_test)
    print(f"train: {train_num}, test: {test_num}")
    print("Training is already done.")
    print(f"image size: {image_size}")
    print()
    
    return (x_test, t_test)

def init_pretrained_network():
    with open("ch3. Neural Network/sample_weight.pkl", 'rb') as f:
        # A file containing pre-trained weight parameters.
        # Predicting directly without training.
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 1-layer
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    # 2-layer
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # 3-layer
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y



# load, init network
(x_test, t_test) = load_dataset()
network = init_pretrained_network()


# # (no batch) predict, accuracy
# accuracy_count = 0
# for i in range(len(x_test)):
#     y = predict(network, x_test[i])
#     p = np.argmax(y)
#     if p == t_test[i]:
#         accuracy_count += 1

# print("accuracy:" , accuracy_count / x_test.shape[0])


# (batch) predict, accuracy
accuracy_count = 0
batch_size = 128
for i in range(0, len(x_test), batch_size):
    y_batch = predict(network, x_test[i:i+batch_size])
    p = np.argmax(y_batch, axis=1)
    accuracy_count += np.sum(p == t_test[i:i+batch_size])

print("accuracy:" , accuracy_count / x_test.shape[0])
