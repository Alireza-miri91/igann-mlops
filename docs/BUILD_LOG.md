# igann-mlops — Build Log (first-hand record)

**Purpose:** a factual, first-hand record of what was actually built, measured, broken, and fixed while making this project. It is the single source of truth for the later deliverables — the `bank_experiments` file, the public README, the LinkedIn post, and the CV skill updates — so that everything we write is backed by real evidence, **not** reconstructed from memory or invented.

**Rule:** only record things that actually happened in this project. Numbers, commands, commit hashes, and outputs are copied from real runs. If something is not yet done, it is marked TODO, not described as done.

_Last updated: 2026-09-10._

---

## Environment (real)
- Machine: macOS (Darwin 25.6), Apple Silicon.
- Python: 3.10.15.
- Virtualenv: `~/.venvs/igann-mlops` (moved OUT of the iCloud-synced project folder — see Lessons #1).
- Git: repo at `portfolio/igann-mlops`, default branch `main`.
- Dataset: scikit-learn California housing, snapshotted to `data/california.csv` (20,640 rows × 9 cols: 8 features + `MedHouseVal` target). Committed to the repo for reproducibility.
- Feature means (from the shipped CSV): MedInc 3.871, HouseAge 28.639, AveRooms 5.429, AveBedrms 1.097, Population 1425.477, AveOccup 3.071, Latitude 35.632, Longitude -119.570, MedHouseVal(target) 2.069.

---

## Bricks completed

### 1. Reproducible training — DONE
- `src/make_dataset.py` snapshots the dataset to `data/california.csv`.
- `src/train.py`: `Pipeline([StandardScaler(set_output=pandas), IGANNRegressor(random_state=42)])`, 80/20 train/test split (`random_state=42`), metrics printed, model saved to `models/model.joblib`.
- Reproducibility pinned four ways: code (git), dependencies (pinned `requirements.txt` via `pip freeze`), data (committed CSV snapshot), randomness (fixed seed 42).
- **Real metrics:**
  - First local run (default `n_estimators`, ~5000 w/ early stopping): **RMSE 0.641, R² 0.686**.
  - Google Colab cross-check (default): RMSE 0.650, R² 0.678 (tiny cross-machine float differences — same seed, different torch build).
  - Final committed model (`n_estimators=3000`): **RMSE 0.652, R² 0.676**.
- Commits: `cb39927` (project setup, gitignore, pinned deps), `77fae4f` (data loader + training script).

### 2. Experiment tracking with MLflow — DONE
- `src/train.py` wrapped in an MLflow run: logs params (`model`, `seed`, `test_size`, `n_estimators`), metrics (`rmse`, `r2`), and the model artifact.
- Tracking store pinned: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`; experiment `igann-california`.
- MLflow UI viewed at **http://localhost:5001** (port 5000 blocked by macOS — see Lessons #3).
- Commit: `c99d433`.

### 3. FastAPI interpretable serving — DONE
- `src/serve.py`: `/health` + `/predict`. `/predict` returns prediction, baseline (average-house) prediction, and per-feature **contributions**.
- Interpretability method: contribution-vs-average-house (swap one feature to its dataset mean, re-predict, difference = that feature's effect). Exact for an additive model like IGANN; model-agnostic (uses only `.predict`, so it survives a future model swap).
- **Real test outputs (served identically from container later):**
  - Bay Area block (MedInc 8.3, …): prediction ≈ **2.69** (~$269k).
  - Inland block (MedInc 2.5, …): prediction **2.3034**, baseline 2.0738; contributions summed exactly to prediction − baseline (0.2296).
  - Inland block (MedInc 2.0, …): prediction **0.5117**, baseline 2.0799; MedInc contribution −1.81 (low income dragging price down).
- Commit: `cde838e`.

### 4. Docker containerization — DONE
- `Dockerfile` (base `python:3.10-slim`) + `.dockerignore`.
- CPU-only PyTorch (`torch==2.13.0+cpu` via `--extra-index-url https://download.pytorch.org/whl/cpu`) → image CONTENT SIZE **~470 MB** (the default CUDA build would have been ~5–6 GB and filled Docker's disk).
- Built `igann-mlops:latest`, ran with `-p 8000:8000`, verified `/predict` from inside the container.
- Commit: `3c339ff`.

### 5. Kubernetes — STAGE 1 (local) DONE
- Local single-node cluster via Docker Desktop (Kubeadm provisioner, shares Docker's image store so the local image runs with `imagePullPolicy: IfNotPresent`).
- `k8s/deployment.yaml` (2 replicas) + `k8s/service.yaml` (`type: LoadBalancer`, localhost:8000).
- Verified: 2 pods Running behind one Service; **self-healing demonstrated** — deleted a pod, Kubernetes recreated a replacement within seconds while staying at 2.
- Commit: "Add Kubernetes Deployment and Service manifests" (hash to confirm).

---

## Bricks remaining
- **5. Stage 2 — AKS on Azure (chosen cloud):** Terraform → Resource Group + ACR + AKS; push image to ACR; `kubectl apply` the SAME manifests; hit the live public IP; then `terraform destroy`. **IN PROGRESS.**
  - Tooling installed: azure-cli 2.90.0, terraform 1.16.1; `az login` OK.
  - Terraform provider azurerm v4.81.0. `terraform init/plan/apply` workflow working.
  - ✅ Resource Group `igann-mlops-rg` (germanywestcentral) created via Terraform — `provisioningState: Succeeded` (2026-09-10). First Azure artifact.
  - ✅ ACR `igannmlopsacr` (Basic tier) created via Terraform (2026-09-10).
  - Lesson: Mac (Apple Silicon) builds arm64 images; AKS default nodes are amd64 → must build/push a `linux/amd64` image or pods crash with "exec format error".
  - ✅ amd64 image built with `docker buildx --platform linux/amd64` and pushed to ACR (`az acr repository list` shows `igann-mlops`) (2026-09-10).
  - ⏸ PAUSED 2026-09-10; resumed 2026-09-11. Resume runbook: `docs/RESUME_AZURE.md`.
  - Lesson (2026-09-11): subscription blocks B-series VMs in germanywestcentral (common on free subs). Azure returns the allowed SKU list in the 400 error. Used `Standard_D2s_v7` (2 vCPU) instead of `Standard_B2s`.
  - ✅ AKS cluster `igann-mlops-aks` created via Terraform (2 resources: cluster + AcrPull role), node pool 1× Standard_D2s_v7, Kubernetes v1.35.7 (2026-09-11).
  - ✅ Deployed the same k8s manifests to AKS (image switched to `igannmlopsacr.azurecr.io/igann-mlops:latest`): 2 pods Running in ~17s (pulled from ACR via the AcrPull role → confirms image + permissions correct).
  - ✅ Service got a real public IP **48.201.224.233:8000** — model served live on the internet from Azure. Evidence saved in `docs/evidence/`.
  - Note: plain HTTP, single node, `terraform destroy` after demo. Future work: HTTPS/ingress, autoscaling.
  - ✅ Verified live: `/predict` at the public IP returned a valid response (contributions summed exactly to prediction − baseline) — model served from AKS.
  - ✅ `terraform destroy` — "Destroy complete! Resources: 4 destroyed" (RG + ACR + AKS + role). Spend back to $0; cluster was up ~1h (a few cents). **Brick #5 (AKS) + #6 (Terraform IaC) DONE.**
- **6. Terraform IaC** (part of stage 2 above). **TODO.**
- **7. GitHub Actions CI/CD** (test → build → push to ACR → deploy to AKS). **TODO.**
- **8. Monitoring: Prometheus + PSI/KS drift monitor.** **TODO.**
- **9. Promote:** public README, `bank_experiments` file, LinkedIn post, CV skill updates. **TODO (explicitly deferred by Ali until the end).**

## NEXT SESSION — resume here: swap in Ali's constrained IGANN (his real research)
Currently the project uses the stock `IGANNRegressor` (package). Replace it with Ali's own model.
- Source: `portfolio/RA_Master_project/Thesis/model_1_1_with_dec_and_inc.py` → `class IGANN`. It's a fork of IGANN adding **monotonicity constraints** via **CVXPY**: pass `monotonicity={"MedInc": +1, "AveOccup": -1, ...}` (dict keyed by real feature NAME; +1 increasing, -1 decreasing) to force each shape function's direction. API: `IGANN(task="regression", monotonicity={...}, random_state=42).fit(X_df, y).predict(X_df)`.
- Plan: (1) copy that file into `src/constrained_igann.py`; (2) add `cvxpy` to requirements + re-freeze; (3) in `train.py`, import his `IGANN` and use it in place of `IGANNRegressor` with a `monotonicity` dict; (4) retrain; (5) verify R² (expect a small dip vs 0.68 — the interpretability trade) and that the earlier nonsensical low-MedInc→+contribution anomaly is fixed; (6) rebuild image → then bricks #7 CI/CD, #8 monitoring.
- Proposed monotonic priors (PENDING Ali's confirmation): `MedInc +1`, `AveRooms +1`, `AveOccup -1`; leave HouseAge/Latitude/Longitude/Population/AveBedrms unconstrained.
- Gotcha to handle: joblib pickles the model by its module name, so `constrained_igann` must be importable the SAME way in `train.py`, `serve.py`, and inside Docker (keep `src/` on the path consistently), or `joblib.load` in serving will fail.

### ✅ DONE 2026-09-14 — constrained IGANN integrated as the model
- His model = monotonicity-constrained IGANN (CVXPY), copied to `src/constrained_igann.py`. Constraints chosen interactively (saved to `configs/monotonicity.json`): `{MedInc:+1, AveRooms:+1, AveOccup:-1}`.
- Result: **RMSE 0.708, R² 0.618** (below stock 0.68 — the honest cost of enforced monotonicity; the constrained booster also converges in ~3 rounds then diverges on this data, so it runs shallow). Saved to models/model.joblib + logged to MLflow.
- Five fixes to make the research code production-ready (write-up gold):
  1. **Target scaling** via `TransformedTargetRegressor` — the model assumes standardized `y`.
  2. **`get_params` fix** — added `monotonicity` so sklearn `clone()` (used by the target wrapper) preserves the constraints.
  3. **Solver swap** — constrained solves were unstable/failing on OSQP; switched to `prob.solve(solver=cp.CLARABEL)`.
  4. **Stabilized training** — `boost_rate=0.05`, `elm_alpha=5.0`, `early_stopping=3–5` to stop the boosting diverging.
  5. **Picklability** — replaced a dynamic `type("LinearModel",...)` with a real module-level `LinearModel` class.
- Interactive constraint elicitation added to `train.py` (`get_monotonicity`): prompts the domain expert, saves to config, offers keep-or-revise on later runs; non-interactive (Docker/CI) reuses the saved config.
- ✅ Serving updated: `serve.py` adds `src` to sys.path so joblib can unpickle `constrained_igann`; `/predict` returns the interpretable contributions and they visibly respect the monotonic constraints (rich house MedInc +1.72; poor house MedInc −0.77 — the earlier low-income anomaly is fixed).
- ✅ Docker image rebuilt with the constrained model + cvxpy; container serves identically to local (prediction 4.0606, MedInc +1.7183). Red herring resolved: earlier "wrong" container outputs were a **stale container still bound to port 8000**, not a model/build problem (verified model file checksum identical + direct `docker run` predict matched local).
- ✅ Brick #7 CI (2026-09-17): repo pushed to GitHub (`Alireza-miri91/igann-mlops`, private) via `gh`; `.github/workflows/ci.yml` runs on every push — install deps → retrain constrained model (non-interactive, uses committed `configs/monotonicity.json`) → check artifact → `docker build`. First run green (4m28s). (Minor: Node 20 deprecation warning on checkout@v4/setup-python@v5 — bump to @v5/@v6 to silence.)
- ✅ Brick #7 CD (2026-09-17): workflow now also logs in to GHCR and pushes the image (`ghcr.io/alireza-miri91/igann-mlops`, tags `latest` + commit SHA) on pushes to main; `permissions: packages: write` + `GITHUB_TOKEN`, lowercased owner. Green. Action versions bumped (checkout@v5, setup-python@v6); only remaining warning is docker/login-action@v3 Node20 (cosmetic).
- ✅ Brick #8 monitoring (2026-09-17): `serve.py` exposes Prometheus metrics at `/metrics` (predictions_total counter, prediction_value_100k + predict_latency_seconds histograms, feature_psi gauge) and a `/drift` endpoint computing per-feature PSI vs the training reference (10 quantile bins, flag > 0.2). `scripts/simulate.py` fires traffic. Validated: 300 normal requests → all PSI 0.01–0.06, no drift; forced MedInc/AveRooms → PSI 12.4, both flagged. Lesson: PSI needs an adequate window — 40 samples gave false positives (~0.3), 300 settled near 0.
- **All 8 build bricks DONE.** Remaining: #9 promote (README + CI badge, bank_experiments file, LinkedIn post, CV skills update, make repo public).

---

## Lessons / real errors and fixes (gold for the LinkedIn post — all genuinely hit)
1. **iCloud-synced `.venv` made imports crawl.** Project lives under `~/Documents` (iCloud). Importing scipy/sklearn took minutes because iCloud fights the thousands of package files. Fix: put the venv on non-synced local disk (`~/.venvs/igann-mlops`).
2. **Default PyTorch pulled gigabytes of NVIDIA CUDA libraries**, filling Docker's virtual disk → "read-only file system" build failure. Fix: CPU-only torch build.
3. **MLflow UI returned HTTP 403 on port 5000** — macOS AirPlay Receiver occupies port 5000. Fix: `mlflow ui --port 5001`.
4. **MLflow "no runs shown"** — writer and reader pointed at different stores. Fix: pin `mlflow.set_tracking_uri("sqlite:///mlflow.db")` and launch the UI with the matching `--backend-store-uri`.
5. **`train_test_split` unpacking order** — returns `X_train, X_test, y_train, y_test` (train/test grouped per array), not X/Y grouped.
6. **Local image in Kubernetes** needs `imagePullPolicy: IfNotPresent`, else k8s tries to pull `:latest` from the internet.
7. Small syntax gotchas (recorded for honesty): pydantic typo, Dockerfile needing `COPY src .` (trailing dot) and a space after `CMD`, and `--extra-index-url` (add) vs `--index-url` (replace) for the torch CPU wheel.

---

## Azure proof to capture (for public repo + LinkedIn) — checklist for stage 2
Collect these as real evidence during the AKS demo and save screenshots under `docs/evidence/`:
- [ ] `terraform apply` output (resources created).
- [ ] Azure Portal screenshot: Resource Group showing the AKS cluster + ACR.
- [ ] Azure Portal screenshot: AKS cluster overview (node count, K8s version, region).
- [ ] Terminal: `kubectl get nodes` pointed at AKS (shows Azure node names/VM sizes).
- [ ] Terminal: `kubectl get pods,svc` on AKS with the Service's **public EXTERNAL-IP**.
- [ ] Browser: the live public URL `/docs` + a real `/predict` result.
- [ ] `az acr repository list` showing the pushed `igann-mlops` image.
- [ ] Azure Cost Management screenshot (proof of real, small spend) — optional but strong.
- [ ] `terraform destroy` output (responsible teardown — good story for the post).

---

## Deferred to the end (per Ali — do NOT do now)
- Create the `bank_experiments` file in the project directory (distilled, first-hand experiment record built from this log).
- Update CV versions (`cv/v2.0/_generator/` data + `system/experience_bank.md`) with the earned skills, once the project is complete.
- Draft the LinkedIn post (German market; GitHub link in first comment) from this log.
- Make the git repo public (after a cleanup pass).
