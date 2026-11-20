import numpy as np

from edge_health.data import make_clients
from edge_health.experiment import evaluate
from edge_health.federated import train_fedavg
from edge_health.heavy_experiment import run


def test_client_generation_is_reproducible():
    first = make_clients(samples_per_client=40, seed=3)
    second = make_clients(samples_per_client=40, seed=3)
    assert np.allclose(first[0]["x_train"], second[0]["x_train"])
    assert np.array_equal(first[0]["y_test"], second[0]["y_test"])


def test_federated_loss_decreases():
    clients = make_clients(samples_per_client=120, seed=5)
    _, _, history, transmitted = train_fedavg(clients, rounds=8)
    assert history[-1]["loss"] < history[0]["loss"]
    assert transmitted > 0


def test_report_metrics_are_valid():
    report, history = evaluate(7)
    assert 0 <= report["federated_global"]["auroc"] <= 1
    assert 0 <= report["centralized"]["accuracy"] <= 1
    assert len(report["per_client_federated"]) == 5
    assert len(history) == 20


def test_heavy_benchmark_is_complete():
    rows, report = run()
    assert len(rows) == 10
    assert report["protocol"]["clients"] == 8
    assert report["primary_comparison"]["relative_reduction"] > 0.5
