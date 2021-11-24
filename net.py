import numpy as np


class Network:
    def __init__(self, layers):
        """Default network constructor that creates network according to layers"""
        self.weights = []
        self.biases = []
        self.inner_potentials = [] # z = W.x + b
        self.outputs = [] # o(z)
        self.layers = layers
        c = 0.1 # Constant to multiply weights with
        for i in range(len(layers) - 1):
            input_size = layers[i][0]
            output_size = layers[i + 1][0]

            # Initialize weights and biases randomly
            self.weights.append(np.random.randn(input_size, output_size) * c)
            self.biases.append(np.random.rand(output_size, 1) * c)

            # Inner potentials and outputs to None for further use
            self.inner_potentials.append(None)
            self.outputs.append(None)

    def __init__(self, layers, weights, biases):
        """Constructor to test the network with initialized weights and biases"""
        self.weights = weights
        self.biases = biases
        self.layers = layers
        self.inner_potentials = [] # z = W.x + b
        self.outputs = [] # o(z)
        for i in range(len(layers)):
            # Inner potentials and outputs to None for further use
            self.inner_potentials.append(None)
            self.outputs.append(None)

    def feedforward(self, X_in):
        o_prev = X_in
        for i in range(len(self.layers)):
            act_f = self.layers[i][1]
            W = self.weights[i]
            b = self.biases[i]
            # z = W.x + b (x is the vector of inputs to the neuron (outputs of previous layer))
            z = np.matmul(W, o_prev) + b 
            o = act_f(z)
            o_prev = o

            self.inner_potentials[i] = z
            self.outputs[i] = o

    def backprop(self):
        ...
    def backprop_layer(self):
        ...
    def get_error(T, O):
        ...

    def train(X, Y, learning_rate, momentum=0):
        ...


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
    net = Network(arch, XOR["weights"], XOR["biases"])
    net.feedforward(np.array([0, 1]).reshape(2, 1))
    net.feedforward(np.array([1, 1]).reshape(2, 1))
    net.feedforward(np.array([1, 0]).reshape(2, 1))
    net.feedforward(np.array([0, 0]).reshape(2, 1))


if __name__ == "__main__":
    np.random.seed(10)
    xor()
