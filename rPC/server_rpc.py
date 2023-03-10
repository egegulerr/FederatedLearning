import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    grpc_max_message_length=1024 * 1024 * 1024,
)
