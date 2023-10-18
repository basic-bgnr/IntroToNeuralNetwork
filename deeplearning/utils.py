import struct
import gzip
import tqdm
import numpy as np
import pathlib


def loadData(
    image_file_location,
    label_file_location,
):
    read_integer = lambda file_object: struct.unpack(">i", file_object.read(4))[0]
    read_image = lambda file_object, row, column: file_object.read(row * column)
    read_byte = lambda file_object: struct.unpack("B", file_object.read(1))[0]

    datas = []

    with gzip.open(image_file_location, "rb") as training_images_file, gzip.open(
        label_file_location, "rb"
    ) as training_labels_file:
        magic_number_images, magic_number_labels = read_integer(
            training_images_file
        ), read_integer(training_labels_file)
        number_images, number_labels = read_integer(training_images_file), read_integer(
            training_labels_file
        )

        image_row, image_column = read_integer(training_images_file), read_integer(
            training_images_file
        )

        assert magic_number_images == 2051, magic_number_labels == 2049
        assert number_images == number_labels
        assert image_row == image_column

        for i in tqdm.tqdm(range(number_images)):
            image_data, image_label = read_image(
                training_images_file, image_row, image_column
            ), read_byte(training_labels_file)
            #         image_data = array(list(image_data)).reshape(image_row, image_column)
            datas.append((image_data, image_label))

    return datas, image_row, image_column


def getDataOutput(y):
    y_vector = np.zeros((10, 1))
    y_vector[y] = 1
    return y_vector


def getStructuredData(image_file_location, label_file_location):
    datas, row, column = loadData(image_file_location, label_file_location)
    normalized_data = []
    for image, label in datas:
        v = np.array(list(image)).reshape((row * column, 1))
        v_norm = v / np.linalg.norm(v)
        normalized_data.append((v_norm, getDataOutput(label)))

    return normalized_data


def getTrainingValidationTestingData(data_set_location):
    training_image_location = f"{data_set_location}train-images-idx3-ubyte.gz"
    training_label_location = f"{data_set_location}train-labels-idx1-ubyte.gz"

    testing_image_location = f"{data_set_location}t10k-images-idx3-ubyte.gz"
    testing_label_location = f"{data_set_location}t10k-labels-idx1-ubyte.gz"

    total_training_data = getStructuredData(
        training_image_location, training_label_location
    )
    training_data = total_training_data[0:60_000]
    validation_data = total_training_data[60_000:]

    testing_data = getStructuredData(testing_image_location, testing_label_location)
    return (training_data, validation_data, testing_data)
