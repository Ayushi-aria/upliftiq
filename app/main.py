import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.conformal import prediction_sets
from src.train import FEATURES

MODELS_DIR = Path("models")
API_KEY = os.getenv("UPLIFTIQ_API_KEY", "dev-key")
VERSION = os.getenv("MODEL_VERSION", "v1")

app = FastAPI(title="UpliftIQ", version="2.0")
api_key_header = APIKeyHeader(name="X-API-Key")

MODEL = joblib.load(MODELS_DIR / VERSION / "model.joblib")
CACHE = {}


def check_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ScoreRequest(BaseModel):
    tenure: float = Field(ge=0, le=50)
    activity: float = Field(ge=-10, le=10)
    risk_score: float = Field(ge=-10, le=10)
    age: int = Field(ge=18, le=100)
    balance: float = Field(ge=0)


class ScoreResponse(BaseModel):
    model_version: str
    uplift: float
    p_control: float
    p_treated: float
    outcome_set_control: list[int]
    outcome_set_treated: list[int]
    confident: bool
    cached: bool


def compute(x: pd.DataFrame) -> dict:
    out = {"uplift": float(MODEL["xl"].effect(x).ravel()[0])}
    for t, name in ((0, "control"), (1, "treated")):
        a = MODEL["arms"][t]
        p = float(a["iso"].predict(a["clf"].predict_proba(x)[:, 1])[0])
        h0, h1 = prediction_sets(np.array([p]), a["qhat"])
        out[f"p_{name}"] = p
        out[f"outcome_set_{name}"] = [c for c, h in ((0, h0[0]), (1, h1[0])) if h]
    out["confident"] = (len(out["outcome_set_control"]) == 1
                        and len(out["outcome_set_treated"]) == 1)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_version": VERSION}


@app.get("/models", dependencies=[Depends(check_key)])
def list_models():
    result = []
    for d in sorted(MODELS_DIR.iterdir()):
        f = d / "metrics.json"
        if f.exists():
            result.append({"version": d.name, "active": d.name == VERSION,
                           "metrics": json.loads(f.read_text())})
    return result


@app.post("/score", response_model=ScoreResponse,
          dependencies=[Depends(check_key)])
def score(req: ScoreRequest):
    key = (VERSION, *req.model_dump().values())
    cached = key in CACHE
    if not cached:
        x = pd.DataFrame([req.model_dump()])[FEATURES]
        CACHE[key] = compute(x)
    return ScoreResponse(model_version=VERSION, cached=cached, **CACHE[key])
class OptimizeUser(ScoreRequest):
    user_id: int


class OptimizeRequest(BaseModel):
    budget: float = Field(gt=0)
    cost_per_user: float = Field(gt=0)
    users: list[OptimizeUser] = Field(min_length=1, max_length=100_000)


@app.post("/optimize", dependencies=[Depends(check_key)])
def optimize(req: OptimizeRequest):
    df = pd.DataFrame([u.model_dump() for u in req.users])
    df["uplift"] = MODEL["xl"].effect(df[FEATURES]).ravel()

    max_targets = int(req.budget // req.cost_per_user)
    positive = df[df["uplift"] > 0]            # sleeping dogs ko kabhi target nahi karte
    chosen = positive.nlargest(max_targets, "uplift")

    n = len(chosen)
    expected = float(chosen["uplift"].sum())
    best_possible = float(positive["uplift"].sum())
    random_expected = float(df["uplift"].mean() * n)

    return {
        "model_version": VERSION,
        "n_users": len(df),
        "n_targeted": n,
        "budget_used": round(n * req.cost_per_user, 2),
        "expected_benefit": round(expected, 3),
        "benefit_if_random_targeting": round(random_expected, 3),
        "pct_of_max_achievable": round(100 * expected / best_possible, 1) if best_possible > 0 else 0.0,
        "targets": [{"user_id": int(r.user_id), "uplift": round(float(r.uplift), 4)}
                    for r in chosen.itertuples()],
    }