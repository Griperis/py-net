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

        self.weights_sum = []
        self.biases_sum = []

        for i in range(len(layers) - 1):

            input_size = layers[i][0]
            output_size = layers[i + 1][0]

            # Initialize weights and biases randomly
            init_method = layers[i][3]
            if init_method is None:
                init_method = '0'

            self.weights.append(self.init_weights(output_size, input_size, init_method))
            self.biases.append(self.init_weights(output_size, 1, init_method))

            self.weights_sum.append(self.init_weights(output_size, input_size, '0'))
            self.biases_sum.append(self.init_weights(output_size, 1, '0'))

            # Inner potentials and outputs to None for further use
            self.inner_potentials.append(None)
            self.outputs.append(None)
            self.deltas.append(np.zeros((output_size, 1)))
            self.prev_deltas.append(np.zeros((output_size, 1)))

    def init_weights(self, output_size, input_size, method):
        if method == 'Xa':
            return np.random.randn(output_size, input_size) * np.sqrt(1/input_size)
        elif method == 'He':
            return np.random.randn(output_size, input_size) * np.sqrt(2/input_size)
        elif method == 'rnd':
            return np.random.uniform(-1, 1, (output_size, input_size))
        elif method == '0':
            return np.zeros((output_size, input_size))
        elif method == '1':
            return np.ones((output_size, input_size))
        else:
            raise RuntimeError("Unsupported initialization method")

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
    
    def update_weights(self, img, learning_rate, momentum):
        for i in range(len(self.weights)):
            output = img if i == 0 else self.outputs[i - 1]
            delta = self.deltas[i]
            prev_delta = self.prev_deltas[i]
            self.weights[i] += -learning_rate * np.matmul(delta, np.transpose(output))
            self.biases[i] += -learning_rate * delta

    def add_gradients(self, img, learning_rate, weights_sum, biases_sum):
        for i in range(len(self.weights)):
            output = img if i == 0 else self.outputs[i - 1]
            delta = self.deltas[i]
            weights_sum[i] += learning_rate * np.matmul(delta, np.transpose(output))
            biases_sum[i] += learning_rate * delta
    
    def update_weights_batch(self, weights_sum, biases_sum, batch_size):
        for i in range(len(self.weights)):
            weights_sum[i] = weights_sum[i] / batch_size
            biases_sum[i] = biases_sum[i] / batch_size
            self.weights[i] -= weights_sum[i]
            self.biases[i] -= biases_sum[i]

    def zero_out(self, weights_sum, biases_sum):
        for i in range(len(self.weights)):
            for arr in weights_sum[i]:
                for element in arr:
                    element = 0.0

            for arr in biases_sum[i]:
                for element in arr:
                    element = 0.0

    def predict(self, img):
        return int(np.argmax(self.feedforward(img)))

    def split_train_set(self, X, Y, split):
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        Y = Y[indices]
        validation_size = X.shape[0] * split
        train_size = int(X.shape[0] - validation_size)
        return (X[:train_size], Y[:train_size], X[train_size:], Y[train_size:])

    def train(self, X, Y, epochs, batch_size=32, learning_rate=0.008, momentum=0.0, validation_split=0.0):

        for epoch in range(epochs):
            train_X, train_Y, val_X, val_Y = self.split_train_set(X, Y, validation_split)

            indices = list(range(0,train_X.shape[0]))
            np.random.shuffle(indices)
            batch_indices = np.array_split(indices, train_X.shape[0] / batch_size)

            for i,batch in enumerate(batch_indices):
                #print(f"Batch {i}/{len(batch_indices)}")
                self.zero_out(self.weights_sum, self.biases_sum)
                for index in batch:
                    img = train_X[index]
                    label = train_Y[index]
                    img.shape += (1,)
                    label.shape += (1,)

                    output = self.feedforward(img)
                    one_hot_truth = one_hot(label)
                    self.backprop(output, one_hot_truth)
                    self.add_gradients(img, learning_rate, self.weights_sum, self.biases_sum)
                
                self.update_weights_batch(self.weights_sum, self.biases_sum, batch_size)
                    
            nr_correct = 0
            for v_img, v_label in zip(val_X, val_Y):
                v_img.shape += (1,)
                v_label.shape += (1,)
                output = self.feedforward(v_img)
                nr_correct += int(np.argmax(output) == np.argmax(one_hot(v_label)))

            print(f"Train acc [{epoch}]: {round((nr_correct / val_X.shape[0]) * 100, 2)}%")
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
        (784, None, None, None),
        (128, sigmoid, sigmoid_d, 'Xa'),
        (10, softmax, sigmoid_d, 'Xa')
    ]

    net = Network(arch)
    (x_train, y_train), (x_test, y_test) = [(x.reshape((len(x), 784)).astype(float)/255, K.utils.to_categorical(y)) for x, y in K.datasets.fashion_mnist.load_data()]
    net.train(x_train, y_train, 64, learning_rate=0.08, validation_split=0.1)
    
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
