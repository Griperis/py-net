from data import get_mnist
import numpy as np
import sys  
import matplotlib.pyplot as plt
from tensorflow import keras as K

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
(x_train, y_train), (x_test, y_test) = [(x.reshape((len(x), 784)).astype(float)/255, K.utils.to_categorical(y)) for x, y in K.datasets.fashion_mnist.load_data()]
images = x_train
labels = y_train
w_i_h = np.random.uniform(-0.5, 0.5, (128, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 128))
b_i_h = np.zeros((128, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 5
for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

test_correct = 0
for img, l in zip(x_test, y_test):
    img.shape += (1,)
    l.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))
    test_correct += int(np.argmax(o) == np.argmax(l))

print(f"Test acc: {round((test_correct / x_test.shape[0]) * 100, 2)}%")