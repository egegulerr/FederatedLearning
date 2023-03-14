import grpc
import numpy as np
import message_pb2_grpc
import message_pb2
from model import Model
from time import sleep

CLIENT_ID = 1


def main():
    with grpc.insecure_channel(
        "0.0.0.0:50052",
        options=[
            ("grpc.max_message_length", 1024 * 1024 * 100),
            ("grpc.max_receive_message_length", 1024 * 1024 * 100),
            ("grpc.max_send_message_length", 1024 * 1024 * 100),
        ],
    ) as channel:
        stub = message_pb2_grpc.FederatedLearningStub(channel)

        model, trained_data_length = Model().initialize()
        for _ in range(5):
            weights = model.get_weights()
            tensor_list = [
                message_pb2.Tensor(data=weight.tobytes(), shape=weight.shape)
                for weight in weights
            ]
            tensors = message_pb2.TensorList(tensors=tensor_list)

            _ = stub.SendWeightsToServer(
                message_pb2.ClientMessage(
                    client_id=CLIENT_ID,
                    data_length=trained_data_length,
                    list_tensors=tensors,
                )
            )

            while True:
                sleep(1)
                response = stub.IsWeightsReady(message_pb2.Empty(value=1))

                if response.ready:
                    break

            response = stub.GetAvgWeightsFromServer(message_pb2.Empty(value=1))
            weights = [
                np.ndarray(tensor.shape, np.float32, np.frombuffer(tensor.data))
                for tensor in response.tensors
            ]
            model.set_weights(weights)
            model, trained_data_length = Model().train_model(model)


if __name__ == "__main__":
    main()
