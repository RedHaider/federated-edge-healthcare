from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import make_clients
from .federated import local_update, sigmoid, train_fedavg


def _metrics(x, y, weight, bias):
    probability = sigmoid(x @ weight + bias)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, probability >= 0.5)),
    }


def evaluate(seed: int = 27):
    clients = make_clients(seed=seed)
    federated_weight, federated_bias, history, transmitted = train_fedavg(clients)
    all_train_x = np.vstack([client["x_train"] for client in clients])
    all_train_y = np.concatenate([client["y_train"] for client in clients])
    all_test_x = np.vstack([client["x_test"] for client in clients])
    all_test_y = np.concatenate([client["y_test"] for client in clients])

    centralized_weight, centralized_bias = local_update(
        all_train_x, all_train_y, np.zeros(all_train_x.shape[1]), 0.0, epochs=60
    )
    local_metrics = []
    client_metrics = []
    for client in clients:
        local_weight, local_bias = local_update(
            client["x_train"], client["y_train"], np.zeros(all_train_x.shape[1]), 0.0, epochs=60
        )
        local_metrics.append(_metrics(client["x_test"], client["y_test"], local_weight, local_bias))
        client_metrics.append({
            "client_id": client["client_id"],
            **_metrics(client["x_test"], client["y_test"], federated_weight, federated_bias),
        })

    client_aurocs = [item["auroc"] for item in client_metrics]
    report = {
        "local_only_mean": {
            key: float(np.mean([item[key] for item in local_metrics])) for key in ("auroc", "accuracy")
        },
        "federated_global": _metrics(all_test_x, all_test_y, federated_weight, federated_bias),
        "centralized": _metrics(all_test_x, all_test_y, centralized_weight, centralized_bias),
        "federated_client_auroc_gap": float(max(client_aurocs) - min(client_aurocs)),
        "communication_bytes": transmitted,
        "per_client_federated": client_metrics,
    }
    return report, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report, history = evaluate(args.seed)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "loss"])
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
