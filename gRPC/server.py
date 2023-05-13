import pickle
import grpc
import message_pb2_grpc
import message_pb2
from concurrent import futures
import numpy as np
import redis
from codecarbon import EmissionsTracker

redix = redis.Redis(host="localhost", port=6379)
print("Redix databse started")


class FederatedLearningServicer(message_pb2_grpc.FederatedLearningServicer):
    NUMBER_OF_CLIENTS = 2

    def IsWeightsReady(self, request, context):
        return message_pb2.IsReady(ready=True if redix.exists("new_weights") else False)

    def SendWeightsToServer(self, request, context):
        client_id = request.client_id
        trained_data_length = request.data_length
        tensors_list = request.list_tensors.tensors

        self.increment_client_count()
        redix.delete("new_weights")
        client_count = int(redix.get("count").decode())

        if client_count <= self.NUMBER_OF_CLIENTS:
            tensors = [
                np.ndarray(tensor.shape, np.float32, np.frombuffer(tensor.data))
                for tensor in tensors_list
            ]
            redix.set(client_id, pickle.dumps(tensors))
            redix.set(f"{client_id}_length", trained_data_length)

        if client_count == self.NUMBER_OF_CLIENTS:
            redix.set("count", 0)

            weights, data_lengths = self.prepare_weights()
            new_weights = self.aggregate_weights(weights, data_lengths)
            redix.set("new_weights", pickle.dumps(new_weights))

        return message_pb2.Empty(value=1)

    @staticmethod
    def aggregate_weights(weights_dict, data_point_legths):
        total_data_points = sum(data_point_legths)
        for index in range(len(weights_dict)):
            scalar = data_point_legths[index] / total_data_points
            for n_array_index in range(len(weights_dict[index])):
                weights_dict[index][n_array_index] *= scalar
        new_weights = [sum(x) for x in zip(*weights_dict)]
        return new_weights

    def prepare_weights(self):
        weights_bytes = [
            redix.get(str(client_index))
            for client_index in range(self.NUMBER_OF_CLIENTS + 1)
        ]
        filtered_weights_bytes = list(filter(lambda x: x is not None, weights_bytes))
        weights_dict = [pickle.loads(content) for content in filtered_weights_bytes]
        data_point_leghts = [
            int(redix.get(f"{index}_length"))
            for index in range(1, len(weights_dict) + 1)
        ]
        return weights_dict, data_point_leghts

    def GetAvgWeightsFromServer(self, request, context):
        weights = pickle.loads(redix.get("new_weights"))
        tensor_list = [
            message_pb2.Tensor(data=weight.tobytes(), shape=weight.shape)
            for weight in weights
        ]
        return message_pb2.TensorList(tensors=tensor_list)

    @staticmethod
    def increment_client_count():
        count = redix.get("count")
        if count:
            redix.set("count", int(count.decode()) + 1)
        else:
            redix.set("count", 1)


def main():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=5),
        options=[
            ("grpc.max_message_length", 1024 * 1024 * 100),
            ("grpc.max_receive_message_length", 1024 * 1024 * 100),
            ("grpc.max_send_message_length", 1024 * 1024 * 100),
        ],
    )
    message_pb2_grpc.add_FederatedLearningServicer_to_server(
        FederatedLearningServicer(), server
    )

    server.add_insecure_port("[::]:50052")
    server.start()
    print("gRPC Server started")
    server.wait_for_termination()


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()

    main()

    tracker.stop()
