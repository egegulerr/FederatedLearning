# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import message_pb2 as message__pb2


class FederatedLearningStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.IsWeightsReady = channel.unary_unary(
                '/grpcThesis.FederatedLearning/IsWeightsReady',
                request_serializer=message__pb2.Empty.SerializeToString,
                response_deserializer=message__pb2.IsReady.FromString,
                )
        self.GetAvgWeightsFromServer = channel.unary_unary(
                '/grpcThesis.FederatedLearning/GetAvgWeightsFromServer',
                request_serializer=message__pb2.Empty.SerializeToString,
                response_deserializer=message__pb2.TensorList.FromString,
                )
        self.SendWeightsToServer = channel.unary_unary(
                '/grpcThesis.FederatedLearning/SendWeightsToServer',
                request_serializer=message__pb2.ClientMessage.SerializeToString,
                response_deserializer=message__pb2.Empty.FromString,
                )


class FederatedLearningServicer(object):
    """Missing associated documentation comment in .proto file."""

    def IsWeightsReady(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAvgWeightsFromServer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendWeightsToServer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FederatedLearningServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'IsWeightsReady': grpc.unary_unary_rpc_method_handler(
                    servicer.IsWeightsReady,
                    request_deserializer=message__pb2.Empty.FromString,
                    response_serializer=message__pb2.IsReady.SerializeToString,
            ),
            'GetAvgWeightsFromServer': grpc.unary_unary_rpc_method_handler(
                    servicer.GetAvgWeightsFromServer,
                    request_deserializer=message__pb2.Empty.FromString,
                    response_serializer=message__pb2.TensorList.SerializeToString,
            ),
            'SendWeightsToServer': grpc.unary_unary_rpc_method_handler(
                    servicer.SendWeightsToServer,
                    request_deserializer=message__pb2.ClientMessage.FromString,
                    response_serializer=message__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'grpcThesis.FederatedLearning', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FederatedLearning(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def IsWeightsReady(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grpcThesis.FederatedLearning/IsWeightsReady',
            message__pb2.Empty.SerializeToString,
            message__pb2.IsReady.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAvgWeightsFromServer(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grpcThesis.FederatedLearning/GetAvgWeightsFromServer',
            message__pb2.Empty.SerializeToString,
            message__pb2.TensorList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendWeightsToServer(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grpcThesis.FederatedLearning/SendWeightsToServer',
            message__pb2.ClientMessage.SerializeToString,
            message__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
