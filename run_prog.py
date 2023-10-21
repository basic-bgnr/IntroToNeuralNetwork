#!/usr/bin/python3.8
import deeplearning.ML as ML
import deeplearning.utils as utils


training_data, validation_data, testing_data = utils.getTrainingValidationTestingData(
    data_set_location="./mnist_dataset/"
)

net = ML.NeuralNetwork([28 * 28, 30, 10])

net.SGD(
    input_data=training_data,
    epochs=60,
    batch_size=10,
    eta=0.5,
    test_data=testing_data,
    lamda=0.5,
)
