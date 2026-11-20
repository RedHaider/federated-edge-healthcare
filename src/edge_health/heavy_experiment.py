from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data import make_clients
from .federated import loss, sigmoid, train_fedavg

SEEDS = (27, 41, 73, 101, 137)


def _proximal_update(x, y, weight, bias, global_weight, global_bias, epochs=3, rate=.12, mu=.04):
    weight, bias = weight.copy(), float(bias)
    for _ in range(epochs):
        error = sigmoid(x @ weight + bias) - y
        weight -= rate * (x.T @ error / len(y) + mu * (weight - global_weight))
        bias -= rate * (float(error.mean()) + mu * (bias - global_bias))
    return weight, bias


def _quantize(vector):
    scale = max(float(np.max(np.abs(vector))) / 127.0, 1e-12)
    code = np.rint(vector / scale).clip(-127, 127).astype(np.int8)
    return code.astype(float) * scale, code.nbytes + 4


def train_robust(clients, rounds=30, dropout=.20, seed=0):
    rng = np.random.default_rng(seed)
    n_features = clients[0]["x_train"].shape[1]
    weight, bias = np.zeros(n_features), 0.0
    transmitted = 0
    history = []
    for round_index in range(rounds):
        active = [c for c in clients if rng.random() >= dropout]
        if not active: active = [clients[int(rng.integers(len(clients)))]]
        updates, priorities = [], []
        for client in active:
            local_w, local_b = _proximal_update(client["x_train"], client["y_train"], weight, bias,
                                                weight, bias)
            packed, nbytes = _quantize(np.r_[local_w - weight, local_b - bias])
            updates.append(packed)
            priorities.append(len(client["y_train"]) * (1 + loss(client["x_train"], client["y_train"], weight, bias)))
            transmitted += nbytes
        priorities = np.asarray(priorities); priorities /= priorities.sum()
        mean_update = sum(p * u for p, u in zip(priorities, updates))
        weight += mean_update[:-1]; bias += float(mean_update[-1])
        history.append({"round": round_index + 1, "active_clients": len(active)})
    return weight, bias, history, transmitted


def _evaluate(clients, weight, bias):
    client_rows = []
    for client in clients:
        probability = sigmoid(client["x_test"] @ weight + bias)
        client_rows.append({"client": client["client_id"],
                            "auroc": float(roc_auc_score(client["y_test"], probability)),
                            "brier": float(brier_score_loss(client["y_test"], probability))})
    aurocs = [r["auroc"] for r in client_rows]
    return {"mean_client_auroc": float(np.mean(aurocs)), "worst_client_auroc": float(np.min(aurocs)),
            "client_auroc_gap": float(np.max(aurocs)-np.min(aurocs)),
            "mean_client_brier": float(np.mean([r["brier"] for r in client_rows]))}


def run():
    rows = []
    for seed in SEEDS:
        clients = make_clients(n_clients=8, samples_per_client=500, seed=seed)
        bw, bb, _, baseline_bytes = train_fedavg(clients, rounds=30, local_epochs=3)
        rw, rb, history, robust_bytes = train_robust(clients, rounds=30, dropout=.20, seed=seed)
        for model, metrics, nbytes in (("fedavg_float64", _evaluate(clients, bw, bb), baseline_bytes),
                                       ("prox_quantized", _evaluate(clients, rw, rb), robust_bytes)):
            rows.append({"seed": seed, "model": model, "communication_bytes": nbytes,
                         "active_clients_last_round": history[-1]["active_clients"] if model == "prox_quantized" else 8,
                         **metrics})
    summary = {}
    for model in ("fedavg_float64", "prox_quantized"):
        subset = [r for r in rows if r["model"] == model]; summary[model] = {}
        for metric in ("mean_client_auroc", "worst_client_auroc", "client_auroc_gap", "mean_client_brier", "communication_bytes"):
            values = np.array([r[metric] for r in subset], dtype=float); margin = 1.96 * values.std(ddof=1)/np.sqrt(len(values))
            summary[model][metric] = {"mean": float(values.mean()), "ci95_low": float(values.mean()-margin),
                                      "ci95_high": float(values.mean()+margin)}
    base_bytes = summary["fedavg_float64"]["communication_bytes"]["mean"]
    new_bytes = summary["prox_quantized"]["communication_bytes"]["mean"]
    report = {"protocol": {"seeds": list(SEEDS), "clients": 8, "samples_per_client": 500,
                           "rounds": 30, "simulated_dropout": .20, "quantization": "signed int8 + float32 scale"},
              "summary": summary, "primary_comparison": {
                  "metric": "communication reduction with AUROC retention",
                  "baseline_bytes": base_bytes, "proposed_bytes": new_bytes,
                  "relative_reduction": 1-new_bytes/base_bytes,
                  "mean_client_auroc_delta": summary["prox_quantized"]["mean_client_auroc"]["mean"]-
                                               summary["fedavg_float64"]["mean_client_auroc"]["mean"]}}
    return rows, report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("outputs/heavy")); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True); rows,report=run()
    (args.output/"heavy_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (args.output/"per_seed_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(report["primary_comparison"],indent=2))


if __name__ == "__main__": main()
