import joblib

from src.monitor import psi
from src.simulator import simulate
from src.train import FEATURES

model = joblib.load("models/v1/model.joblib")
ref = model["reference"]

fresh = simulate(n=5000, seed=123)
drifted = fresh.copy()
drifted["activity"] += 1.5      # activity ka distribution shift kar do

print(f"{'feature':12s} {'no drift':>10s} {'drifted':>10s}")
for f in FEATURES:
    a = psi(fresh[f].values, ref[f]["edges"], ref[f]["expected"])
    b = psi(drifted[f].values, ref[f]["edges"], ref[f]["expected"])
    print(f"{f:12s} {a:10.4f} {b:10.4f}")