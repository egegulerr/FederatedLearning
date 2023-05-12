from model import create_model, save_model, train_model, compile_model
from protocolls import SerializationTypes


def deploy_model(weights, serialization_method):
    model = create_model()
    compiled_model = compile_model(model)
    compiled_model.set_weights(weights)
    save_model(model, serialization_method)


def aggregate_weights(weights_dict, data_point_legths):
    total_data_points = sum(data_point_legths)
    for index in range(len(weights_dict)):
        scalar = data_point_legths[index] / total_data_points
        for n_array_index in range(len(weights_dict[index])):
            weights_dict[index][n_array_index] *= scalar
    new_weights = [sum(x) for x in zip(*weights_dict)]
    return new_weights



def Main():
    # run_api()
    print("API Server is running")
    print("ML Model is being created and fitted")
    model = create_model()
    model = compile_model(model)
    model, train_data_length = train_model(model)

    serialization_method = input(
        "Which serialization method do you want to use ? \n 1-Protobuffer \n 2-Pickel \n 3-Json"
    )

    if serialization_method == SerializationTypes.JSON.value:
        print("JSON serialization is selected. Serializing the model")
        save_model(model, SerializationTypes.JSON.name)
    elif serialization_method == SerializationTypes.PICKEL.value:
        print("PICKEL serizalization is selected. Serializing the model")
        save_model(model, SerializationTypes.PICKEL.name)
    elif serialization_method == SerializationTypes.PROTOBUFFER.value:
        print("PROTOBUFFER serialization is selected. Serializing the model")
        save_model(model, SerializationTypes.PROTOBUFFER.name)

    input(
        "Waiting for the client to finish its job. \n *****PRES ENTER TO CONTINUE*****"
    )

    print("Getting weights from client")


if __name__ == "__main__":
    Main()
