# UpliftIQ v2: Calibrated Causal Decision Platform

A small, end-to-end system that decides **who should receive an intervention**
(a reminder, an offer, a risk warning) by estimating the *causal effect* of the
intervention per user, with calibrated probabilities and uncertainty sets,
served through a REST API.

> **Everything here runs on simulated data.** The simulator has known true
> treatment effects, so models can be verified against ground truth.
> No real user data and no live deployment results are claimed.

## Why uplift and not risk prediction

A risk score tells you who is likely to have an outcome, not whom the
intervention will change. In the simulator, high-risk users are "sleeping dogs":
the intervention backfires for them.

![Uplift vs budget](docs/uplift_vs_budget.png)

| Targeting top 20% of users by | Share of max achievable benefit |
|---|---|
| X-learner uplift | **78.4%** |
| Risk score | **-54.9%** (net harm) |

## What is in the system

| Part | What it does |
|---|---|
| `src/simulator.py` | Generates 50,000 users with known true uplift, randomized treatment |
| `src/train.py` | X-learner and DR-learner (EconML + LightGBM) vs a risk-score baseline |
| `src/calibrate.py` | Per-arm isotonic calibration, ECE before and after |
| `src/conformal.py` | Split-conformal outcome sets with 90% target coverage |
| `src/train_final.py` | Trains the final pipeline and saves a versioned artifact in `models/<version>/` |
| `src/monitor.py` | PSI feature drift and ECE calibration drift |
| `src/make_plots.py` | Generates the plots and the top-20% business metric used in this README |
| `src/demo_drift.py` | Shifts one feature on purpose to show the drift monitor reacting |
| `app/main.py` | FastAPI service: `/score`, `/optimize`, `/monitor`, `/models`, `/health` |
| `tests/` | Unit, API, and property-based (Hypothesis) tests |
| `locustfile.py` | Load test for the API |
| `.github/workflows/ci.yml` | Lint, train and test on every push |

## Results (simulated data, held-out test set)

- **Ranking:** X-learner rank-correlation with true uplift 0.967, DR-learner 0.941,
  risk-score baseline -0.407.
- **Calibration (control arm):** ECE 0.0325 to 0.0110 after isotonic calibration.
  Calibration improves probability quality; it does not change the ranking.
- **Conformal coverage (target 90%):** 0.913 (treated arm), 0.961 (control arm).
  Control-arm coverage is conservative, likely because isotonic output is
  step-shaped and creates tied scores.

![Reliability diagram](docs/reliability.png)

## API

All endpoints except `/health` need the header `X-API-Key`.

- `POST /score` returns uplift, calibrated probabilities for both arms,
  conformal outcome sets, and a `confident` flag.
- `POST /optimize` takes a budget and a cost per user, and returns the target list
  with expected benefit. Users with non-positive uplift are never targeted.
- `POST /monitor` takes recent records and returns per-feature PSI with a status
  (ok, warning, alert), ECE per arm when outcomes (`T`, `Y`) are supplied, and a
  `retrain_recommended` flag.
- `GET /models` lists model versions and metrics.

Run locally:

```bash
pip install -r requirements.txt
python -m src.simulator
python -m src.train_final v1
uvicorn app.main:app --reload
```

Docs open at `http://127.0.0.1:8000/docs`.

## Drift monitoring

The training-time distribution of every feature (10 quantile bins) is stored inside
the model artifact, so monitoring always compares against the data the model was
trained on. PSI thresholds: below 0.1 ok, 0.1 to 0.25 warning, above 0.25 alert.

Demo (`python -m src.demo_drift`): 5,000 fresh simulated users, then the same users
with `activity` shifted by +1.5.

| Feature | PSI, no drift | PSI, shifted |
|---|---|---|
| tenure | 0.0011 | 0.0011 |
| activity | 0.0019 | **2.0571** |
| risk_score | 0.0025 | 0.0025 |
| age | 0.0024 | 0.0024 |
| balance | 0.0027 | 0.0027 |

Only the shifted feature raises an alert, and unshifted data gives no false alarm.

## Load test (local, single worker)

Locust, 20 concurrent users, 60 seconds, server and load generator on the same
laptop. Not a production benchmark.

| Endpoint | Median | p95 |
|---|---|---|
| `POST /score` (uncached) | 69 ms | 210 ms |
| `POST /score` (cached) | 32 ms | 160 ms |
| `POST /optimize` (200 users per call) | 62 ms | 190 ms |

~2,970 requests, 0 failures, ~50 requests/s sustained. Throughput is set by the
test's think time (0.1 to 0.5 s per user), so it is not the server's maximum.
The cache saves model inference but not HTTP and validation overhead, so its gain
is modest.

Reproduce: `uvicorn app.main:app`, then
`locust -f locustfile.py --headless -u 20 -r 5 -t 60s --host http://127.0.0.1:8000`.

## Engineering notes

- **Versioning and rollback:** each training run writes `models/<version>/`.
  Set `MODEL_VERSION` to serve another version.
- **Validation and auth:** Pydantic input bounds, API key from the
  `UPLIFTIQ_API_KEY` environment variable.
- **Property-based tests:** probabilities stay in [0, 1], the budget is never
  exceeded, and negative-uplift users are never targeted. A Hypothesis test caught
  a real bug: `budget_used` was rounded up and could exceed the budget.
- **Test coverage:** 37% overall (API layer `app/main.py` 83%). Training, plotting
  and demo scripts are not unit-tested; training runs end to end in CI.

## Simplifications and known limitations

- DuckDB, Postgres, Redis, Celery, Docker, and AWS deployment are not
  implemented. The cache is an in-memory dict (single process, no expiry).
- Drift monitoring is on-demand through `/monitor`. There is no scheduled job,
  alerting, or automatic retraining, and the `/monitor` endpoint itself has no
  API-level test (the PSI functions do).
- Conformal sets are for the binary outcome, not for the individual treatment
  effect, which is never observed.
- `uplift` (X-learner) and `p_treated - p_control` (two calibrated models) can
  differ; ranking uses the X-learner, probabilities come from the calibrated models.
- Load test numbers come from a laptop running server and load generator together.
- Results are from a simulator and may not transfer to real data.