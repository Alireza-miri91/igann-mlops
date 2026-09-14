import os
import sys
import json
import joblib
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

sys.path.append(os.path.dirname(__file__))
from constrained_igann import IGANN

SEED = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 50
CONFIG_PATH = "configs/monotonicity.json"


def _elicit(features, current):
    print("\nDomain-expert step: set a monotonic direction per feature.")
    print("  i = increasing, d = decreasing, n = none, Enter = keep current\n")
    label = {1: "increasing", -1: "decreasing"}
    mono = {}
    for feat in features:
        cur = current.get(feat)
        cur_txt = label.get(cur, "none")
        while True:
            choice = input(f"  {feat} [current: {cur_txt}]: [i/d/n/Enter] ").strip().lower()
            if choice in ("", "i", "d", "n"):
                break
            print("    please type i, d, n, or just Enter")
        if choice == "":
            value = cur
        elif choice == "i":
            value = 1
        elif choice == "d":
            value = -1
        else:
            value = None
        if value is not None:
            mono[feat] = value
    return mono


def get_monotonicity(features):
    current = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            current = json.load(f)

    if not sys.stdin.isatty():
        print(f"Non-interactive: using saved constraints {current}")
        return current

    if current:
        print(f"\nExisting constraints: {current}")
        if input("Keep these? [Y/n] ").strip().lower() in ("", "y", "yes"):
            return current

    mono = _elicit(features, current)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(mono, f, indent=2)
    print(f"\nSaved constraints to {CONFIG_PATH}: {mono}\n")
    return mono


df = pd.read_csv("data/california.csv")
X = df.drop(columns=["MedHouseVal"])
Y = df["MedHouseVal"]

MONOTONICITY = get_monotonicity(list(X.columns))

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=SEED
)

base_pipeline = Pipeline([
    ("scaler", StandardScaler().set_output(transform="pandas")),
    ("igann", IGANN(task="regression", n_estimators=N_ESTIMATORS,
                    random_state=SEED, monotonicity=MONOTONICITY, verbose=1,
                    boost_rate=0.05, elm_alpha=5.0, early_stopping=5)),
])

# Your IGANN expects a standardized target; scale y for training and
# automatically un-scale predictions back to $100k units.
model = TransformedTargetRegressor(regressor=base_pipeline, transformer=StandardScaler())

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("igann-california")

with mlflow.start_run():
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    print(f"RMSE: {rmse:.3f} (average error in $100k)")
    print(f"R2: {r2:.3f} (1 = perfect fit)")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    mlflow.log_param("model", "ConstrainedIGANN")
    mlflow.log_param("seed", SEED)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("monotonicity", json.dumps(MONOTONICITY))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_artifact("models/model.joblib")
    print("Logged run to MLflow.")