import requests
import pickle
from model import train_model, compile_model, create_model
from time import sleep

BASE_URL = "http://localhost:8000"
CLIENT_ID = 2


def send_weights(weights):
    pickled_file = pickle.dumps(weights)
    requests.post(
        f"{BASE_URL}/sendweights/pickel",
        data=pickled_file,
        headers={"Content-Type": "application/octet-stream"},
    )

    print(f"Weights are sent as pickle object")


def Main():
    print("Client is running")
    print(f"FASTApi is selected as communication protocol. Getting ML model")
    model = create_model()
    compiled_model = compile_model(model)
    trained_model, trained_data_length = train_model(compiled_model)

    for rounds in range(5):
        weights = model.get_weights()
        pickl_file = pickle.dumps(weights)

        requests.post(
            f"{BASE_URL}/sendweights/pickle/{CLIENT_ID}/{trained_data_length}",
            data=pickl_file,
            headers={"Content-Type": "application/octet-stream"},
        )

        while True:
            sleep(1)
            response = requests.get(f"{BASE_URL}")
            if response.content.decode() == "true":
                print("New weights are ready. Fetching them")
                break

        response = requests.get(f"{BASE_URL}/getavgweights/pickle")

        weights_pickle = pickle.loads(response.content)
        model.set_weights(weights_pickle)
        model, trained_data_length = train_model(model)

    print("Model is trained. Sending it weights to server")
    send_weights(trained_model.get_weights())

    input(
        "Waiting for ML Model from server to process weights. \n *****PRESS ENTER TO CONTINUE*****"
    )


if __name__ == "__main__":
    Main()
