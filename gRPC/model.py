from tensorflow import keras
import tensorflow as tf
import random


class Model:
    def initialize(self):
        model = self.create_model()
        model = self.compile_model(model)
        model, data_length = self.train_model(model)
        return model, data_length

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def compile_model(model):
        adadelta = tf.optimizers.Adadelta()
        model.compile(
            optimizer=adadelta,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train_model(self, model):
        x_train, x_test, y_train, y_test = self.get_train_data()
        epochs = 1
        # batch_size = random.randint(40, 60)
        batch_size = 50
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

        # Evaluate the model
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print("Test Accuracy: {}".format(test_acc))

        return model, epochs * batch_size
