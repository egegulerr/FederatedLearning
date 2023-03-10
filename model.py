import json
import random

import tensorflow as tf
from tensorflow import keras
import pickle
import requests
import numpy as np


class ModelWrapper:
    def __init__(self, model):
        self._model = model

    def __call__(self):
        return self._model


def get_train_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Convert the arrays to float32 and Normalize the features
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Standardize the features
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Reshape the data to add a channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, x_test, y_train, y_test


def compile_model(model):
    adadelta = tf.optimizers.Adadelta()
    model.compile(
        optimizer=adadelta, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(model):
    x_train, x_test, y_train, y_test = get_train_data()
    epochs = 1
    batch_size = random.randint(40, 60)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test Accuracy: {}".format(test_acc))

    return model, epochs * batch_size


def create_model():
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(keras.layers.ReLU())
    model.add(
        keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Softmax())

    # Train the model
    print("Model Created")
    print(model.summary())

    return model


def save_model(model, method: str):
    method = method.lower()

    if method == "json":
        # Convert Model Architecture to JSON
        model_json = model.to_json()
        # Convert Model Weights to JSON
        weights = model.get_weights()

        with open("saved_models/model_json/model_json.json", "w") as json_file:
            json_file.write(model_json)
        print("Model Arc. saved as JSON with name model_json")
        weights_dict = {}
        for i, weight in enumerate(weights):
            weights_dict["weights_" + str(i)] = weight.tolist()

        with open("saved_models/model_json/weights_json.json", "w") as f:
            json.dump(weights_dict, f)
        print("Model weights saved as JSON with name weights_json")

    if method == "protobuffer":
        # Save model and its weights as Protobuffer
        model.save("saved_models/model_protobuffer/protobuffer/")
        print(
            "Model arc and weights saved as Protobuffer in model_protobuffer directory"
        )

    if method == "pickle":
        wraped = ModelWrapper(model)
        weights = model.get_weights()

        with open("saved_models/model_pickel/model.pkl", "wb") as f:
            pickle.dump(wraped, f)
            print("Model arc and weights saved as Pickel in model_pickel directory")

        with open("saved_models/model_pickel/model_weights.pkl", "wb") as f:
            pickle.dump(weights, f)
            print("Model weights saved as Pickel in model_pickel directory")
