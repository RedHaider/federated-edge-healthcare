from __future__ import annotations

import numpy as np


def make_clients(n_clients: int = 5, samples_per_client: int = 400, seed: int = 27):
    """Create non-IID synthetic site datasets with deterministic train/test splits."""
    rng = np.random.default_rng(seed)
    true_weight = np.array([1.1, -0.9, 0.7, 0.5, -0.45, 0.35])
    clients = []
    for client_id in range(n_clients):
        site_shift = rng.normal(scale=0.55, size=true_weight.size)
        x = rng.normal(size=(samples_per_client, true_weight.size)) + site_shift
        site_bias = (client_id - (n_clients - 1) / 2) * 0.18
        logit = x @ true_weight + site_bias + rng.normal(scale=0.55, size=samples_per_client)
        probability = 1 / (1 + np.exp(-logit))
        y = rng.binomial(1, probability).astype(float)
        order = rng.permutation(samples_per_client)
        split = int(samples_per_client * 0.75)
        train, test = order[:split], order[split:]
        clients.append({
            "client_id": client_id,
            "x_train": x[train],
            "y_train": y[train],
            "x_test": x[test],
            "y_test": y[test],
        })
    return clients
