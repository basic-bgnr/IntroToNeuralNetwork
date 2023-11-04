#!/usr/bin/python3.8
import deeplearning.ML as ML
import deeplearning.utils as utils

import pickle
import numpy as np

data_set_location = "./mnist_dataset/"
training_data_ratio = 1.0
validation_data_ratio = 1.0 / 6.0
testing_data_ratio = 1.0


network_layers = [10, 30, 30, 28 * 28]
epochs = 40
batch_size = 10
eta = 0.5
lamda = 0.5
mu = 0.1  # no momentum when mu = 1
dropout_ratio = 0.2

print(f"{'Neural Network Layers ([nodes])':<35}: {network_layers}")
print(
    f"{'Hyper Parameters':<35}: {epochs=}, {batch_size=}, {eta=}, {lamda=}, {mu=}, {dropout_ratio=}\n"
)

training_data, validation_data, testing_data = utils.getTrainingValidationTestingData(
    data_set_location=data_set_location,
    training_data_ratio=training_data_ratio,
    validation_data_ratio=validation_data_ratio,
    testing_data_ratio=testing_data_ratio,
)


def reverse_tuple(lst):
    return list(map(lambda item: tuple(reversed(item)), lst))


training_data = reverse_tuple(training_data)
validation_data = reverse_tuple(validation_data)
testing_data = reverse_tuple(testing_data)


print(
    f"\n{'Training/Validation/Testing Data':<35}: {list(map(len, [training_data, validation_data, testing_data]))} Images Loaded"
)

with open("./final_network/mnist.pkl", "rb") as open_file:
    adverserial_net = pickle.load(open_file)


def eval_func(feedForward, x, y):
    image_output = feedForward(x)
    number_output = adverserial_net.feedForward(image_output)
    return np.argmax(x), np.argmax(number_output)


net = ML.NeuralNetwork(network_layers)
net.SGD(
    input_data=training_data,
    test_data=testing_data,
    epochs=epochs,
    batch_size=batch_size,
    eta=eta,
    lamda=lamda,
    mu=mu,
    dropout_ratio=dropout_ratio,
    eval_function=eval_func,
)

print("\nTraining Completed\n")
result = net.evaluate(validation_data, eval_func)
print(f"{'Validataion':<35}: {result} / {len(validation_data)}")

with open("./final_network/reverse_mnist.pkl", "wb") as open_file:
    pickle.dump(net, open_file)

print(f"Writing data to ./final_network/reverse_mnist.pkl")
