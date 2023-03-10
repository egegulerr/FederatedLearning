import flwr as fl
from model import create_model, get_train_data, compile_model


model = create_model()
model = compile_model(model)

x_train, x_test, y_train, y_test = get_train_data()


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(
            x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0
        )
        hist = r.history
        print(f"Fit history: {hist}")
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Eval Accuracy: {acc}")
        return loss, len(x_test), {"accuracy": float(acc)}


fl.client.start_numpy_client(
    server_address="0.0.0.0:8080",
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024
)
