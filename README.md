# Federated Edge Health Lite

An original, privacy-safe portfolio benchmark exploring distributed machine
learning for edge healthcare.
It simulates several healthcare edge sites that collaboratively train a risk
classifier with Federated Averaging without sharing their raw records.

This project is not a reproduction of a professor's method and is not a
clinical tool. Every record is synthetic.

## Research question

How does federated learning behave when healthcare edge sites have different
patient populations, and what accuracy, fairness, and communication tradeoffs
appear relative to local-only and centralized training?

## Run

```powershell
python -m pip install -e .
python -m edge_health.experiment --output outputs/demo
pytest -q
```

## What is measured

- AUROC and accuracy for local-only, federated, and centralized models
- Per-client federated AUROC and the worst-to-best client performance gap
- Approximate model-update bytes transmitted during federated training
- Round-by-round federated training loss

## Design

Five simulated sites receive non-identically distributed populations. Each
site performs local gradient updates, after which a server averages parameters
in proportion to site sample count. Test data remain local to each site.

## Limitations

The model is intentionally small and educational. It does not provide secure
aggregation, differential privacy, network-failure simulation, or medical
validity. These are clear extensions for a research-grade version.
