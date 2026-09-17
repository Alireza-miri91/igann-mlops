"""FastAPI serving for the constrained IGANN: interpretable /predict, /health, Prometheus /metrics, and PSI /drift."""
import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

sys.path.append(os.path.dirname(__file__))

warnings.filterwarnings("ignore")

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

# Reference (training) data — used for the baseline and for drift
REFERENCE = pd.read_csv("data/california.csv")[FEATURES]
baseline = REFERENCE.mean()

model = joblib.load("models/model.joblib")

app = FastAPI(title="IGANN California Housing")

# --- Prometheus metrics ---
PREDICTIONS = Counter("predictions_total", "Total number of predictions served")
PREDICTION_VALUE = Histogram(
    "prediction_value_100k", "Predicted house value ($100k)",
    buckets=(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5),
)
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Latency of /predict (seconds)")
FEATURE_PSI = Gauge("feature_psi", "Population Stability Index per feature (drift)", ["feature"])

# --- Drift setup: quantile bin edges from the reference data ---
N_BINS = 10
BIN_EDGES = {}
for f in FEATURES:
    edges = np.unique(np.quantile(REFERENCE[f], np.linspace(0, 1, N_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    BIN_EDGES[f] = edges

recent_requests = []  # in-memory buffer of incoming feature rows


def psi(reference_values, current_values, edges):
    eps = 1e-6
    exp = np.histogram(reference_values, bins=edges)[0]
    act = np.histogram(current_values, bins=edges)[0]
    exp = exp / max(exp.sum(), 1) + eps
    act = act / max(act.sum(), 1) + eps
    return float(np.sum((act - exp) * np.log(act / exp)))


class House(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/drift")
def drift():
    if len(recent_requests) < 10:
        return {"status": "not enough requests yet", "n_requests": len(recent_requests)}
    current = pd.DataFrame(recent_requests)
    scores = {}
    for f in FEATURES:
        score = psi(REFERENCE[f].values, current[f].values, BIN_EDGES[f])
        scores[f] = round(score, 4)
        FEATURE_PSI.labels(feature=f).set(score)
    drifted = [f for f, s in scores.items() if s > 0.2]
    return {"n_requests": len(recent_requests), "psi": scores, "drifted_features": drifted}


@app.post("/predict")
def predict(house: House):
    start = time.perf_counter()

    features = house.model_dump()
    recent_requests.append(features)

    row = pd.DataFrame([features])[FEATURES]
    prediction = float(model.predict(row)[0])

    baseline_row = pd.DataFrame([baseline])[FEATURES]
    baseline_prediction = float(model.predict(baseline_row)[0])

    contributions = {}
    for feature in FEATURES:
        tweaked = row.copy()
        tweaked[feature] = baseline[feature]
        tweaked_prediction = float(model.predict(tweaked)[0])
        contributions[feature] = round(prediction - tweaked_prediction, 4)

    PREDICTIONS.inc()
    PREDICTION_VALUE.observe(prediction)
    PREDICT_LATENCY.observe(time.perf_counter() - start)

    return {
        "prediction": round(prediction, 4),
        "unit": "$100,000",
        "baseline_prediction": round(baseline_prediction, 4),
        "contributions": contributions,
    }