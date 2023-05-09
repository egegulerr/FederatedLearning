from enum import Enum


class SerializationTypes(Enum):
    PROTOBUFFER = "1"
    PICKEL = "2"
    JSON = "3"


class CommunicationTypes(Enum):
    GRPC = "1"
    FASTAPI = "2"
