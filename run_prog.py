#!/usr/bin/python3.8
import deeplearning.ML as ML
import deeplearning.utils as utils


data_set_location = "./mnist_dataset/"
training_data_ratio = 1.0
validation_data_ratio = 1.0 / 6.0
testing_data_ratio = 0.1
testing_data_ratio = 1.0


network_layers = [28 * 28, 30, 30, 10]
epochs = 40
batch_size = 10
eta = 0.2
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
training_data = training_data * 2
print(
    f"\n{'Training/Validation/Testing Data':<35}: {list(map(len, [training_data, validation_data, testing_data]))} Images Loaded"
)

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
)
print("\nTraining Completed\n")
print(
    f"{'Validataion':<35}: {net.evaluate(validation_data, ML.eval_function)} / {len(validation_data)}"
)
