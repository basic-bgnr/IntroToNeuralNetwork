import numpy as np
import random


class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)

        consecutive_layer_pair = zip(layers, layers[1:])
        # print consecutive_layer_pair

        self.weights = [np.random.randn(k, j) for j, k in consecutive_layer_pair]
        self.bias = [np.random.randn(i, 1) for i in layers[1:]]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
        # func = lambda x : max(0, x)
        # return np.vectorize(func)(z)

    def sigmoid_prime(self, z):
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)
        # func = lambda x: 1 if x>0 else 0
        # return np.vectorize(func)(z)

    def batch_data(self, data, size):
        return [data[k : k + size] for k in range(0, len(data), size)]

    def SGD(self, input_data, epochs, batch_size, eta, test_data=None):
        for generation in range(epochs):
            random.shuffle(input_data)
            batches = self.batch_data(input_data, batch_size)

            for batch in batches:
                self.trainBatch(batch, eta)

            if test_data:
                n_test_data = len(test_data)
                print(f"Epoch {generation}: {self.evaluate(test_data)} / {n_test_data}")
            else:
                print(f"Epoch {generation} completed")

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, x):
        for w, b in zip(self.weights, self.bias):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def trainBatch(self, batch, eta):
        m = float(len(batch))
        new_weights = [np.zeros(w.shape) for w in self.weights]
        new_bias = [np.zeros(b.shape) for b in self.bias]

        for x, y in batch:
            delta_weight, delta_bias = self.backPropagate(x, y)
            new_weights = [d_w + n_w for d_w, n_w in zip(new_weights, delta_weight)]
            new_bias = [d_b + n_b for d_b, n_b in zip(new_bias, delta_bias)]

        self.weights = [
            w - (eta * 1 / m) * nw for nw, w in zip(new_weights, self.weights)
        ]
        self.bias = [b - (eta * 1 / m) * nb for nb, b in zip(new_bias, self.bias)]

    def backPropagate(self, x, y):
        delta_weights = [None] * (self.num_layers - 1)
        delta_bias = [None] * (self.num_layers - 1)

        activation = x
        activations = [x]
        activation_primes = [self.sigmoid_prime(x)]

        zs = []

        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
            activation_prime = self.sigmoid_prime(z)
            activation_primes.append(activation_prime)

        delta_a = activations[-1] - y

        for l in range(1, self.num_layers):
            delta_bias[-l] = delta_a * activation_primes[-l]
            delta_weights[-l] = np.dot(
                delta_bias[-l], activations[-(l + 1)].transpose()
            )

            delta_a = np.dot(
                self.weights[-l].transpose(), (delta_a * activation_primes[-l])
            )

        return delta_weights, delta_bias
