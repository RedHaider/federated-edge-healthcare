from __future__ import annotations

import numpy as np


def sigmoid(value):
    value = np.clip(value, -30, 30)
    return 1 / (1 + np.exp(-value))


def loss(x, y, weight, bias):
    probability = np.clip(sigmoid(x @ weight + bias), 1e-8, 1 - 1e-8)
    return float(-(y * np.log(probability) + (1 - y) * np.log(1 - probability)).mean())


def local_update(x, y, weight, bias, epochs: int = 3, learning_rate: float = 0.12):
    weight, bias = weight.copy(), float(bias)
    for _ in range(epochs):
        error = sigmoid(x @ weight + bias) - y
        weight -= learning_rate * (x.T @ error / len(y))
        bias -= learning_rate * float(error.mean())
    return weight, bias


def train_fedavg(clients, rounds: int = 20, local_epochs: int = 3, learning_rate: float = 0.12):
    n_features = clients[0]["x_train"].shape[1]
    weight, bias = np.zeros(n_features), 0.0
    history = []
    for round_index in range(rounds):
        updates, counts = [], []
        for client in clients:
            update = local_update(
                client["x_train"], client["y_train"], weight, bias, local_epochs, learning_rate
            )
            updates.append(update)
            counts.append(len(client["y_train"]))
        total = sum(counts)
        weight = sum(w * n for (w, _), n in zip(updates, counts)) / total
        bias = sum(b * n for (_, b), n in zip(updates, counts)) / total
        mean_loss = sum(
            loss(client["x_train"], client["y_train"], weight, bias) * len(client["y_train"])
            for client in clients
        ) / total
        history.append({"round": round_index + 1, "loss": mean_loss})
    parameters = n_features + 1
    communication_bytes = rounds * len(clients) * parameters * 8 * 2
    return weight, bias, history, communication_bytes
