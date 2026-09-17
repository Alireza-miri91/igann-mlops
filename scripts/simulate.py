"""Load-test the running API: replay real houses to /predict, then print the /drift report.

Usage: python scripts/simulate.py [normal|drift]   ('drift' forces MedInc/AveRooms to extremes).
"""
import sys, json, urllib.request
import pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
N = 300
FEATURES = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]

sample = pd.read_csv("data/california.csv").sample(N, random_state=0)

for _, r in sample.iterrows():
    body = {c: float(r[c]) for c in FEATURES}
    if MODE == "drift":          # simulate a shifted world
        body["MedInc"] = 15.0
        body["AveRooms"] = 20.0
    data = json.dumps(body).encode()
    req = urllib.request.Request("http://localhost:8000/predict", data=data,
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req).read()

print(f"sent {N} '{MODE}' requests")
print(urllib.request.urlopen("http://localhost:8000/drift").read().decode())
