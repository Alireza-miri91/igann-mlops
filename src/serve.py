import warnings
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore")

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


#Baseline = an "average house" : the mean of every feature across the dataset
baseline = pd.read_csv("data/california.csv")[FEATURES].mean()

model = joblib.load("models/model.joblib")

app = FastAPI(title = "IGANN California Housing")

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

@app.post("/predict")
def predict(house: House):
    row = pd.DataFrame([house.model_dump()])[FEATURES]
    prediction = float(model.predict(row)[0])
    
    baseline_row = pd.DataFrame([baseline])[FEATURES]
    baseline_prediction = float(model.predict(baseline_row)[0])
    
    contributions = {}
    for feature in FEATURES:
        tweaked = row.copy()
        tweaked[feature] = baseline[feature]
        tweaked_prediction = float(model.predict(tweaked)[0])
        contributions[feature] = round(prediction - tweaked_prediction, 4)
        
    return {
        "prediction": round(prediction, 4) ,
        "unit": "$100,000",
        "baseline_prediction": round(baseline_prediction, 4),
        "contributions": contributions,    
    }



