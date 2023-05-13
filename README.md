# FederatedLearning
Federated Learning System, in which the serialization method and communication protocols can be controlled


# gRPC
In gRPC folder you can find server and client scripts. Connection is established with gRPC and the weights are serialized using Protobuffer.

# FASTAPI
The client, api and server method at the base directory are using FAST API to communicate. Right now JSON is used to serialize weights.

In FASTAPI the state of the variables are not stored between workers. Thats why Redis database is used. Since I am focusing on the energy consumption of the communication protocols and serialization methos.
Redis database is also used in gRPC.
