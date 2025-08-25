import flwr as fl
from flwr.server.strategy import FedAvg

# Dummy evaluation function to enforce client-side evaluation
def evaluate_fn(server_round, parameters, config):
    print(f"📡 Server triggering evaluation at round {server_round}")
    return None

# FedAvg strategy with 3 clients and 40 rounds
strategy = FedAvg(
    evaluate_fn=evaluate_fn,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=40),  # ← 40 rounds
    strategy=strategy,
)
