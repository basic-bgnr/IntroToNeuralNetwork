#!/usr/bin/python3.8
import deeplearning.ML as ML
import deeplearning.utils as utils


data_set_location = "./mnist_dataset/"
training_data_ratio = 1.0
validation_data_ratio = 1.0 / 6.0
testing_data_ratio = 0.10


network_layers = [28 * 28, 30, 10]
epochs = 30
batch_size = 10
eta = 2.0
lamda = 0.9
mu = 0.3

training_data, validation_data, testing_data = utils.getTrainingValidationTestingData(
    data_set_location=data_set_location,
    training_data_ratio=training_data_ratio,
    validation_data_ratio=validation_data_ratio,
    testing_data_ratio=testing_data_ratio,
)

print(
    f"\n{'Training/Validation/Testing Data':<35}: {list(map(len, [training_data, validation_data, testing_data]))} Images Loaded"
)


print(f"{'Neural Network Layers ([nodes])':<35}: {network_layers}")
print(f"{'Hyper Parameters':<35}: {epochs=}, {batch_size=}, {eta=}, {lamda=}, {mu=}\n")

net = ML.NeuralNetwork(network_layers)
net.SGD(
    input_data=training_data,
    test_data=testing_data,
    epochs=epochs,
    batch_size=batch_size,
    eta=eta,
    lamda=lamda,
    mu=mu,
)
print("\nTraining Completed\n")
print(f"{'Validataion':<35}: {net.evaluate(validation_data)} / {len(validation_data)}")
