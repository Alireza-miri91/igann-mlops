# igann-mlops — deploying an interpretable, constraint-aware ML model end to end

![CI](https://github.com/Alireza-miri91/igann-mlops/actions/workflows/ci.yml/badge.svg)

Production MLOps around **my own research model**: a *constrained IGANN* — an interpretable Generalized Additive Neural Network whose per-feature effects can be forced monotone (increasing/decreasing) from domain-expert priors. The project takes it from a trained model to a live, self-explaining, monitored web service, with the full lifecycle automated.

Dataset: California housing (predict median house value). The model is the point, not the dataset.

## Why it's interesting
- **Interpretable by construction.** Every prediction is decomposed into per-feature contributions against an "average house" — you see *why*, not just *what*.
- **Domain knowledge is enforced, not hoped for.** A domain expert sets monotonic directions (e.g. *income can only push price up*); the model guarantees them via a CVXPY-constrained fit. This fixes real anomalies the unconstrained model produced (e.g. below-average income raising the predicted price).
- **Research code made production-grade.** Target scaling, sklearn-clone compatibility, a robust QP solver (CLARABEL), training stabilization, and picklability — the unglamorous work that turns a thesis model into something deployable.

## Stack / architecture
`data snapshot → train (constrained IGANN) → MLflow tracking → FastAPI service → Docker → Kubernetes (local + Azure AKS) → Terraform IaC → GitHub Actions CI/CD → Prometheus + drift monitoring`

- **Training** — scikit-learn `Pipeline` (`StandardScaler` → `IGANN`) wrapped in `TransformedTargetRegressor`; monotonicity constraints elicited interactively and saved to `configs/monotonicity.json` for reproducible / CI runs.
- **Tracking** — MLflow logs params, metrics, and the model artifact.
- **Serving** — FastAPI: `/predict` (prediction + interpretable contributions), `/health`, `/metrics` (Prometheus), `/drift` (per-feature PSI).
- **Packaging** — Docker (CPU-only PyTorch, ~470 MB image).
- **Orchestration** — Kubernetes manifests; deployed to Azure AKS provisioned entirely with Terraform (resource group + ACR + cluster), then torn down.
- **CI/CD** — GitHub Actions: on every push it reinstalls, **retrains from source**, verifies the artifact, builds the image, and publishes it to GHCR.
- **Monitoring** — Prometheus metrics + a PSI data-drift detector that flags features whose incoming distribution shifts from the training data.

## Results
- **R² ≈ 0.62, RMSE ≈ 0.71** ($100k units) on the held-out test set. The constrained model trades a little raw fit for guaranteed-sensible, monotone explanations — an intentional, honest tradeoff.
- **Constraints hold in production.** With `MedInc +`, `AveRooms +`, `AveOccup −`: a high-income house yields a positive `MedInc` contribution, a low-income house a negative one — never the reverse.
- **Drift detection works.** On ~300 real requests all feature PSI stays ≈ 0.01–0.06 (no drift); forcing `MedInc`/`AveRooms` to extremes drives their PSI far past the 0.2 threshold and flags them.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/make_dataset.py     # snapshot the dataset (once)
python src/train.py            # elicit constraints, train, log to MLflow
uvicorn src.serve:app --port 8000   # serve at http://localhost:8000/docs
```

Run in a container:
```bash
docker build -t igann-mlops .
docker run -p 8000:8000 igann-mlops
```

## API
| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /predict` | Prediction + per-feature contributions vs an average house |
| `GET /metrics` | Prometheus metrics (counts, latency, prediction distribution, drift PSI) |
| `GET /drift` | Per-feature PSI vs the training data; flags drifted features |

## Repo layout
```
src/          constrained_igann.py (model), train.py, serve.py, make_dataset.py
configs/      monotonicity.json  (the expert's saved constraint choices)
k8s/          Kubernetes Deployment + Service
terraform/    AKS + ACR + resource group as code
.github/      CI/CD workflow
scripts/      simulate.py  (traffic generator for the drift demo)
docs/         BUILD_LOG.md  (first-hand build record)
```

## Notes & future work
- The constrained booster converges in a few rounds on this data, so the fitted model is deliberately shallow; the emphasis is trustworthy interpretability over squeezing out R².
- Serving is plain HTTP; production would add TLS/ingress and autoscaling.
- Possible next steps: Grafana dashboards over the Prometheus metrics, and automated retraining triggered by drift alerts.
