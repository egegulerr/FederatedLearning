import json
import numpy as np
import requests
import pickle
from model import ModelWrapper, train_model, compile_model, create_model
from protocolls import SerializationTypes
from time import sleep

BASE_URL = "http://localhost:8000"
CLIENT_ID = 2


def send_weights(weights):
    weights_dict = {}
    for i, weight in enumerate(weights):
        weights_dict["weights_" + str(i)] = weight.tolist()
    json_file = json.dumps(weights_dict)

    requests.post(f"{BASE_URL}/sendweights/client1", json=json_file)


    print(f"Weights are sent as {SerializationTypes.JSON.name} object")


def Main():
    print("Client is running")

    model = create_model()
    compiled_model = compile_model(model)
    trained_model, trained_data_length = train_model(compiled_model)
    for rounds in range(5):
        weights = model.get_weights()
        weights_dict = {}
        for i, weight in enumerate(weights):
            weights_dict["weights_" + str(i)] = weight.tolist()
        json_file = json.dumps(weights_dict)

        requests.post(
            f"{BASE_URL}/sendweights/json/{CLIENT_ID}/{trained_data_length}",
            json=json_file,
        )

        while True:
            sleep(1)
            response = requests.get(f"{BASE_URL}")
            if response.content.decode() == "true":
                print("New weights are ready. Fetching them")
                break

        response = requests.get(f"{BASE_URL}/getavgweights/json")

        weights_json = json.loads(json.loads(response.content))
        weights = [np.array(w) for w in weights_json.values()]
        model.set_weights(weights)
        model, trained_data_length = train_model(model)

    print("Model is trained. Sending it weights to server")
    send_weights(trained_model.get_weights(), SerializationTypes.JSON.value)


if __name__ == "__main__":
    Main()
