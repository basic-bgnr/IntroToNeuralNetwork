import numpy as np
import random
import tqdm


class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers

        consecutive_layer_pair = zip(self.layers, self.layers[1:])
        # print consecutive_layer_pair

        self.weights = [
            np.random.randn(k, j) / j**0.5 for j, k in consecutive_layer_pair
        ]
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

    def batchData(self, data, size):
        return [data[k : k + size] for k in range(0, len(data), size)]

    def SGD(
        self,
        input_data,
        epochs,
        batch_size,
        eta,
        test_data=None,
        lamda=0.0,
        mu=1.0,
        dropout_ratio=0.0,
    ):
        n_train_data = len(input_data)
        n_test_data = len(test_data) if test_data else test_data

        # for generation in tqdm.tqdm(range(epochs), desc="Total Progress", leave=False):
        for generation in range(epochs):
            random.shuffle(input_data)
            batches = self.batchData(input_data, batch_size)

            momentum_v = [np.zeros(w.shape) for w in self.weights]
            for batch in tqdm.tqdm(
                batches,
                desc=f"{'Processing Minibatch':<35}",
                leave=False,
            ):
                self.trainBatch(
                    batch, eta, lamda, n_train_data, momentum_v, mu, dropout_ratio
                )

            epoch_generation = f"Epoch {generation}"
            if test_data:
                print(
                    f"{epoch_generation:<35}: {self.evaluate(test_data)} / {n_test_data}"
                )
            else:
                print(f"{epoch_generation} completed")

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []
        for x, y in tqdm.tqdm(test_data, desc=f"{'Evaluating':<35}", leave=False):
            test_results.append((np.argmax(self.feedForward(x)), np.argmax(y)))

        return sum(int(x == y) for (x, y) in test_results)

    def feedForward(self, x):
        for w, b in zip(self.weights, self.bias):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def trainBatch(
        self, batch, eta, lamda, n_train_data, momentum_v, mu, dropout_ratio
    ):
        m = float(len(batch))
        new_weights = [np.zeros(w.shape) for w in self.weights]
        new_bias = [np.zeros(b.shape) for b in self.bias]

        # in this dropout filter we don't touch the input image layer and the output layer
        # i.e. we only filter the hidden layers
        # the following dropout filter is appended with [(output_neuron_num, 1)] so that it wouldn't filter
        # output layer, it might seem that since the last layer is filled with ones so as to not modify the output
        # layer but why isn't the first layer is also added, for this we need to look at the backpropagation algorith
        # where multiplication with filter matrix is calculated out ( basically the first neuron is left as it is before looping)

        dropout_filter = (
            # [
            #     np.random.binomial(1.0, (1.0 - dropout_ratio), (self.layers[0], 1))
            #     / (1.0 - dropout_ratio)
            # ]
            # + [
            [
                np.random.binomial(1.0, (1.0 - dropout_ratio), b.shape)
                / (1.0 - dropout_ratio)
                for b in new_bias[:-1]  # avoid the output layer
            ]
            + [np.ones((self.layers[-1], 1))]
        )  #
        for x, y in batch:
            delta_weight, delta_bias = self.backPropagateCrossEntropy(
                x, y, dropout_filter
            )
            new_weights = [d_w + n_w for d_w, n_w in zip(new_weights, delta_weight)]
            new_bias = [d_b + n_b for d_b, n_b in zip(new_bias, delta_bias)]

        for index, (n_w, v_w, w) in enumerate(
            zip(new_weights, momentum_v, self.weights)
        ):
            momentum_v[index] = ((1 - mu) * v_w) - (
                eta * (n_w / m + lamda * w / n_train_data)
            )

        self.weights = [(w + v_w) for v_w, w in zip(momentum_v, self.weights)]
        self.bias = [b - (eta * 1 / m) * nb for nb, b in zip(new_bias, self.bias)]

    def backPropagateMeanSquared(self, x, y, dropout_filter):
        delta_weights = [None] * (self.num_layers - 1)
        delta_bias = [None] * (self.num_layers - 1)

        activation = x
        activations = [x]  # * dropout_filter[0]
        activation_primes = [self.sigmoid_prime(x)]

        zs = []

        for w, b, f in zip(
            self.weights, self.bias, dropout_filter
        ):  # dropout_filter[:-1]):
            z = np.dot(w, activation) + b
            z = z
            zs.append(z)
            activation = self.sigmoid(z) * f
            activations.append(activation)
            activation_prime = self.sigmoid_prime(z) * f
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

    # def backPropagateCrossEntropy(self, x, y):
    def backPropagateCrossEntropy(self, x, y, dropout_filter):
        delta_weights = [None] * (self.num_layers - 1)
        delta_bias = [None] * (self.num_layers - 1)

        activation = x  # * dropout_filter[0]
        activations = [x]
        activation_primes = [self.sigmoid_prime(x)]

        zs = []

        for w, b, f in zip(
            self.weights, self.bias, dropout_filter
        ):  # dropout_filter[1:]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z) * f
            activations.append(activation)
            activation_prime = self.sigmoid_prime(z) * f
            activation_primes.append(activation_prime)

        delta_a = -(y / activations[-1] - (1 - y) / (1 - activations[-1]))
        delta_bias[-1] = activations[-1] - y
        delta_weights[-1] = np.dot(delta_bias[-1], activations[-2].transpose())

        for l in range(2, self.num_layers):
            delta_a = np.dot(
                self.weights[-(l - 1)].transpose(),
                (delta_a * activation_primes[-(l - 1)]),
            )

            delta_bias[-l] = delta_a * activation_primes[-l]
            delta_weights[-l] = np.dot(
                delta_bias[-l], activations[-(l + 1)].transpose()
            )

        return delta_weights, delta_bias
