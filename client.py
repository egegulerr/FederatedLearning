import json
import numpy as np
import keras.models
import requests
import pickle
from model import ModelWrapper, train_model, compile_model, create_model
from protocolls import SerializationTypes, CommunicationTypes
from time import sleep

BASE_URL = "http://localhost:8000"
CLIENT_ID = 1


def send_weights(weights, method):
    if method == SerializationTypes.PICKEL.value:
        method = SerializationTypes.PICKEL.name
        pickled_file = pickle.dumps(weights)
        requests.post(
            f"{BASE_URL}/sendweights/pickel",
            data=pickled_file,
            headers={"Content-Type": "application/octet-stream"},
        )

    if method == SerializationTypes.JSON.value:
        method = SerializationTypes.JSON.name
        weights_dict = {}
        for i, weight in enumerate(weights):
            weights_dict["weights_" + str(i)] = weight.tolist()
        json_file = json.dumps(weights_dict)

        requests.post(f"{BASE_URL}/sendweights/client1", json=json_file)

    if method == "protobuffer":
        pass

    print(f"Weights are sent as {method} object")


def initialize_model(serialization_method, model_raw, weights_raw=None):
    if serialization_method == SerializationTypes.JSON.name:
        json_weights_dict = json.loads(weights_raw.content)
        weights = [np.array(w) for w in json_weights_dict.values()]

        model = keras.models.model_from_json(model_raw.content)
        model.set_weights(weights)

    if serialization_method == SerializationTypes.PICKEL.name:
        # TODO Weightse ihtiyac yok
        pickel_model = pickle.loads(model_raw.content)
        model = pickel_model()

    if serialization_method == SerializationTypes.PROTOBUFFER.name:
        pass

    return model


def get_model_with_fastapi(serialization_method):
    if serialization_method == SerializationTypes.JSON.value:
        serialization_method = SerializationTypes.JSON.name
        response_model = requests.get("http://127.0.0.1:8000/getmodel/json")
        print(
            f"Got the serialized model as {SerializationTypes.JSON.name} object. Getting the weigths now."
        )
        response_weights = requests.get("http://127.0.0.1:8000/getweights/json")
        print(
            f"Got the model weights as {SerializationTypes.JSON.name} object. Initializing model"
        )

        model = initialize_model(serialization_method, response_model, response_weights)

    elif serialization_method == SerializationTypes.PICKEL.value:
        serialization_method = SerializationTypes.PICKEL.name
        response_model = requests.get("http://127.0.0.1:8000/getmodel/pickel")
        print(
            f"Got the serialized model as {SerializationTypes.PICKEL.name} object. Getting the weigths now."
        )
        response_weights = requests.get("http://127.0.0.1:8000/getweights/pickel")
        print(
            f"Got the model weights as {SerializationTypes.PICKEL.name} object. Initializing model"
        )

        model = initialize_model(serialization_method, response_model, response_weights)
    elif serialization_method == SerializationTypes.PROTOBUFFER.value:
        pass

    return model


# def Main():
#     print("Client is running")
#     input(
#         "Please wait until ML model is created. \n*****PRESS ENTER WHEN MODEL IS CREATED*****"
#     )
#
#     serialization_method = input(
#         "Which serialization method was used to save the ML model ? \n 1-Protobuffer \n 2-Pickel \n 3-Json"
#     )
#
#     communication_protocoll = input(
#         "Which communication protocoll would you like to use? \n 1-gRPC \n 2-FastAPI"
#     )
#
#     if communication_protocoll == CommunicationTypes.FASTAPI.value:
#         print(
#             f"{CommunicationTypes.FASTAPI.name} is selected as communication protocol. Getting ML model"
#         )
#         model = get_model_with_fastapi(serialization_method)
#         print("Got the ML model. Training it 2 epochs")
#         compiled_model = compile_model(model)
#         trained_model = train_model(compiled_model)
#
#         print("Model is trained. Sending it weights to server")
#         send_weights(trained_model.get_weights(), serialization_method)
#
#     else:
#         pass
#
#     input(
#         "Waiting for ML Model from server to process weights. \n *****PRESS ENTER TO CONTINUE*****"
#     )


def Main():
    print("Client is running")
    input(
        "Please wait until ML model is created. \n*****PRESS ENTER WHEN MODEL IS CREATED*****"
    )

    serialization_method = input(
        "Which serialization method was used to save the ML model ? \n 1-Protobuffer \n 2-Pickel \n 3-Json"
    )

    communication_protocoll = input(
        "Which communication protocoll would you like to use? \n 1-gRPC \n 2-FastAPI"
    )

    if communication_protocoll == CommunicationTypes.FASTAPI.value:
        print(
            f"{CommunicationTypes.FASTAPI.name} is selected as communication protocol. Getting ML model"
        )
        # model = get_model_with_fastapi(serialization_method)
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
        send_weights(trained_model.get_weights(), serialization_method)

    else:
        pass

    input(
        "Waiting for ML Model from server to process weights. \n *****PRESS ENTER TO CONTINUE*****"
    )


if __name__ == "__main__":
    Main()
