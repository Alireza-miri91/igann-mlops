import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
from igann import IGANNRegressor

SEED = 42

df = pd.read_csv('data/california.csv')

X = df.drop(columns=["MedHouseVal"])
Y = df["MedHouseVal"]

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2, random_state=SEED
    )

pipeline = Pipeline([
    ("scaler", StandardScaler().set_output(transform="pandas")),
    ("igann", IGANNRegressor(random_state=SEED)),
])

pipeline.fit(X_train, Y_train)

predictions = pipeline.predict(X_test)
rmse = root_mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

print(f"RMSE: {rmse:.3f} (average error in $100k)")
print(f"R2: {r2:.3f} (1 = perfect fit)")

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model.joblib")
print("Model saved to models/model.joblib")










