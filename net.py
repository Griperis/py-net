import numpy as np
import time
from tensorflow import keras as K # Used for fashion mnist dataset

def one_hot(v):
    o = np.zeros(v.shape[0])
    o[np.argmax(v)] = 1.0
    o.shape += (1,)
    return o


class Network:
    def __init__(self, layers):
        """Default network constructor that creates network according to layers"""
        self.weights = []
        self.biases = []
        self.inner_potentials = [] # z = W.x + b
        self.outputs = [] # o(z)
        self.deltas = []
        self.prev_deltas = [] # used for momentum
        self.layers = layers
        c = 0.5 # Constant to multiply random weights with
        for i in range(len(layers) - 1):
            input_size = layers[i][0]
            output_size = layers[i + 1][0]

            # Initialize weights and biases randomly
            self.weights.append(np.random.randn(output_size, input_size) * c)
            self.biases.append(np.random.rand(output_size, 1) * c)

            # Inner potentials and outputs to None for further use
            self.inner_potentials.append(None)
            self.outputs.append(None)
            self.deltas.append(np.zeros((output_size, 1)))
            self.prev_deltas.append(np.zeros((output_size, 1)))

    def set_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def feedforward(self, x_in):
        o_prev = x_in
        for i in range(len(self.layers) - 1):
            act_f = self.layers[i + 1][1]
            W = self.weights[i]
            b = self.biases[i]
            # z = W.x + b (x is the vector of inputs to the neuron (outputs of previous layer))
            z = np.matmul(W, o_prev) + b 
            o = act_f(z)
            o_prev = o

            self.inner_potentials[i] = z
            self.outputs[i] = o

        return self.outputs[-1]

    def backprop(self, o, t):
        self.prev_deltas = self.deltas.copy() # store deltas before updating them for momentum
        self.deltas[-1] = self.error_d(o, t) # output - target (target should be one hot value)
        if len(self.layers) == 2:
            return

        # E. g. when having 3 layers of weights we want to have indices 1, 0
        # this requires range(3 - 2, -1, -1), -2 -> -1 for last layer and -1 because range is inclusive 
        for i in range(len(self.weights) - 2, -1, -1):
            act_func_d = self.layers[i + 1][2]
            prev_delta = self.deltas[i + 1]
            prev_z = self.inner_potentials[i]
            W = self.weights[i + 1]
            delta_hidden = np.matmul(np.transpose(W), prev_delta) * act_func_d(prev_z)
            self.deltas[i] = delta_hidden 

    def error(self, output, truth, eps=1e-12):
        #return -np.sum(truth*np.log(np.clip(output, eps, 1. - eps) + 1e-9)) / output.shape[0]
        return 1 / len(output) * np.sum((output - truth) **2)
    
    def error_d(self, output, truth):
        return output - truth
    
    def update_weights(self, img, learning_rate, momentum=0.5):
        for i in range(len(self.weights)):
            output = img if i == 0 else self.outputs[i - 1]
            delta = self.deltas[i]
            prev_delta = self.prev_deltas[i]
            self.weights[i] += -learning_rate * np.matmul(delta + momentum * prev_delta, np.transpose(output))
            self.biases[i] += -learning_rate * delta + momentum * prev_delta

    def predict(self, img):
        return int(np.argmax(self.feedforward(img)))

    def train(self, X, Y, epochs, learning_rate=0.008, momentum=0.5):
        nr_correct = 0
        for epoch in range(epochs):
            for img, label in zip(X, Y):
                img.shape += (1,)
                label.shape += (1,)
                output = self.feedforward(img)
                one_hot_truth = one_hot(label)
                # TODO: training set prediction (this is just to validate accuracy)
                nr_correct += int(np.argmax(output) == np.argmax(one_hot_truth))
                self.backprop(output, one_hot_truth)
                self.update_weights(img, learning_rate, momentum)
        
            print(f"Train data acc [{epoch}]: {round((nr_correct / X.shape[0]) * 100, 2)}%")
            nr_correct = 0


def relu(V):
    return np.maximum(V, 0)


def relu_d(V):
    V[V<=0] = 0.0
    V[V>0] = 1.0
    return V


def sigmoid(V):
    return 1 / (1 + np.exp(-V))


def sigmoid_d(V):
    return (sigmoid(V) * (1 - sigmoid(V)))


def softmax(v):
    # Numerically stable with large exponentials
    exps = np.exp(v - v.max())
    return exps / np.sum(exps, axis=0)


def tanh(V):
    return np.tanh(V)

def tanh_d(V):
    return 1.0 / (np.cosh(V)*np.cosh(V))


def xor():
    def step(V):
        return np.where(V >= 0, 1, 0)
    
    XOR = {
        "weights": [
            np.array([[2, 2], [-2, -2]]),
            np.array([1, 1])
        ],
        "biases": [
            np.array([-1, 3]).reshape(2, 1),
            np.array([-2])
        ]
    }
    arch = [
        (2, step, tanh_d), # act functions are ambiguous for first layer
        (1, step, tanh_d)
    ]
    net = Network(arch)
    net.set_weights(XOR["weights"], XOR["biases"])
    net.feedforward(np.array([0, 1]).reshape(2, 1))
    net.feedforward(np.array([1, 1]).reshape(2, 1))
    net.feedforward(np.array([1, 0]).reshape(2, 1))
    net.feedforward(np.array([0, 0]).reshape(2, 1))


def fashion():
    s_time = time.time()
    np.random.seed(5)
    arch = [
        # Set input functions for first layer to None to have sensible error
        (784, None, None),
        #(128, sigmoid, sigmoid_d),
        (10, softmax, sigmoid_d)
    ]

    net = Network(arch)
    (x_train, y_train), (x_test, y_test) = [(x.reshape((len(x), 784)).astype(float)/255, K.utils.to_categorical(y)) for x, y in K.datasets.fashion_mnist.load_data()]
    net.train(x_train, y_train, 64, 0.08, 0)
    
    nr_correct = 0
    for img, l in zip(x_test, y_test):
        img.shape += (1,)
        l.shape += (1,)
        if int(np.argmax(l)) == net.predict(img):
            nr_correct += 1

    e_time = time.time()
    print(f"Test data acc: {nr_correct/x_test.shape[0]*100:0.2f}%")
    print(f"Execution time {e_time-s_time:0.0f}s")

if __name__ == "__main__":
    np.random.seed(10)
    fashion()
