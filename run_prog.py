#!/usr/bin/python3.8
import deeplearning.AI as AI
import deeplearning.utils as utils


training_data, validation_data, testing_data = utils.getTrainingValidationTestingData(
    data_set_location="./mnist_dataset/"
)

net = AI.NeuralNetwork([28 * 28, 16, 16, 10])

net.SGD(
    input_data=training_data, epochs=30, batch_size=10, eta=4, test_data=testing_data
)
