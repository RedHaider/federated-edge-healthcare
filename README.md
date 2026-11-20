# Federated Edge Health: Communication-Efficient Benchmark

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

## Heavy benchmark

The expanded benchmark evaluates eight non-IID healthcare sites across five
independent seeds. It compares full-precision FedAvg with a proximal,
loss-aware aggregation method using signed int8 update quantization and 20%
simulated client dropout. The report includes mean and worst-client AUROC,
client disparity, Brier score, transmitted bytes, and 95% confidence intervals.

```powershell
python -m edge_health.heavy_experiment --output outputs/heavy
```

The communication and performance claims apply only to this synthetic
simulation. The project does not reproduce or improve a professor's published
system, implement real networking, or provide clinical validation.

### Verified result

Across five seeds, signed-int8 quantization plus simulated client dropout
reduced transmitted update bytes from **26,880** to **2,149** per run, a
**92.0% reduction**. Mean client AUROC was retained (0.83526 vs 0.83549), as
were worst-client AUROC (0.77390 vs 0.77416) and the client AUROC gap (0.11056
vs 0.11051). These tiny performance differences are not presented as accuracy
or fairness improvements; the supported claim is communication reduction
without material degradation in this simulation. Full confidence intervals
are in `results/heavy_metrics.csv`.
