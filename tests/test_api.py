from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from app.main import app

client = TestClient(app)
H = {"X-API-Key": "dev-key"}
GOOD = {"tenure": 2.0, "activity": 1.2, "risk_score": 0.3, "age": 35, "balance": 1.5}


def test_health_is_open():
    assert client.get("/health").status_code == 200


def test_score_rejects_missing_or_wrong_key():
    assert client.post("/score", json=GOOD).status_code in (401, 403)
    r = client.post("/score", json=GOOD, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_score_rejects_invalid_input():
    bad = {**GOOD, "age": 5}
    assert client.post("/score", json=bad, headers=H).status_code == 422


def test_models_endpoint_lists_active_version():
    r = client.get("/models", headers=H)
    assert r.status_code == 200
    assert any(m["active"] for m in r.json())


feature_strategy = st.fixed_dictionaries({
    "tenure": st.floats(0, 50, allow_nan=False),
    "activity": st.floats(-10, 10, allow_nan=False),
    "risk_score": st.floats(-10, 10, allow_nan=False),
    "age": st.integers(18, 100),
    "balance": st.floats(0, 1000, allow_nan=False),
})


@settings(max_examples=30, deadline=None)
@given(feature_strategy)
def test_probabilities_always_in_unit_interval(features):
    r = client.post("/score", json=features, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["p_control"] <= 1.0
    assert 0.0 <= body["p_treated"] <= 1.0


@settings(max_examples=20, deadline=None)
@given(
    users=st.lists(feature_strategy, min_size=1, max_size=25),
    k=st.integers(1, 30),
    cost=st.floats(1, 500, allow_nan=False),
)
def test_budget_is_never_violated(users, k, cost):
    users = [{**u, "user_id": i} for i, u in enumerate(users)]
    budget = k * cost
    r = client.post("/optimize", headers=H,
                    json={"budget": budget, "cost_per_user": cost, "users": users})
    assert r.status_code == 200
    body = r.json()
    assert body["budget_used"] <= budget + 1e-6
    assert body["n_targeted"] <= len(users)
    assert all(t["uplift"] > 0 for t in body["targets"])