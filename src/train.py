import os
import joblib
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
from igann import IGANNRegressor

SEED = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 3000

df = pd.read_csv('data/california.csv')

X = df.drop(columns=["MedHouseVal"])
Y = df["MedHouseVal"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=TEST_SIZE, random_state=SEED
    )

pipeline = Pipeline([
    ("scaler", StandardScaler().set_output(transform="pandas")),
    ("igann", IGANNRegressor(random_state=SEED, n_estimators=N_ESTIMATORS)),
])

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("igann-california")

with mlflow.start_run():
    pipeline.fit(X_train, Y_train)

    predictions = pipeline.predict(X_test)
    rmse = root_mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)

    print(f"RMSE: {rmse:.3f} (average error in $100k)")
    print(f"R2: {r2:.3f} (1 = perfect fit)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/model.joblib")

    mlflow.log_param("model","IGANNRegressor")
    mlflow.log_param("seed", SEED)
    mlflow.log_param("test_size",TEST_SIZE)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_artifact("models/model.joblib")
    
    print("Logged run to MLflow.")









