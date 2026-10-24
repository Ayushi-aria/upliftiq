import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from econml.metalearners import XLearner
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.conformal import ALPHA, conformal_threshold, prediction_sets
from src.monitor import build_reference
from src.train import FEATURES, lgbm, top_k_gain


def fit_arm(train, iso_set, conf, t):
    """Ek arm (T=t) ka outcome model + isotonic calibrator + conformal qhat."""
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                         num_leaves=31, verbose=-1)
    tr = train[train["T"] == t]
    clf.fit(tr[FEATURES], tr["Y"])

    i = iso_set[iso_set["T"] == t]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(clf.predict_proba(i[FEATURES])[:, 1], i["Y"])

    c = conf[conf["T"] == t]
    pc = iso.predict(clf.predict_proba(c[FEATURES])[:, 1])
    qhat = conformal_threshold(pc, c["Y"].values, ALPHA)
    return {"clf": clf, "iso": iso, "qhat": float(qhat)}


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"

    df = pd.read_csv("data/sim.csv")
    train, rest = train_test_split(df, test_size=0.5, random_state=0)
    iso_set, rest = train_test_split(rest, test_size=0.7, random_state=0)
    conf, test = train_test_split(rest, test_size=0.5, random_state=0)

    # Ranking ke liye X-learner
    xl = XLearner(models=lgbm(), cate_models=lgbm(),
                  propensity_model=LogisticRegression())
    xl.fit(train["Y"], train["T"], X=train[FEATURES])

    # Probability + conformal ke liye har arm ka calibrated model
    arms = {t: fit_arm(train, iso_set, conf, t) for t in (0, 1)}

    # Test set pe metrics
    true = test["true_uplift"].values
    pred = xl.effect(test[FEATURES]).ravel()
    metrics = {
        "version": version,
        "n_rows": len(df),
        "top20_true_uplift": round(float(top_k_gain(pred, true)), 4),
        "random_true_uplift": round(float(true.mean()), 4),
    }
    for t in (0, 1):
        te = test[test["T"] == t]
        a = arms[t]
        p = a["iso"].predict(a["clf"].predict_proba(te[FEATURES])[:, 1])
        has0, has1 = prediction_sets(p, a["qhat"])
        covered = np.where(te["Y"].values == 1, has1, has0)
        metrics[f"coverage_T{t}"] = round(float(covered.mean()), 3)

    out = Path("models") / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "xl": xl,
            "arms": arms,
            "features": FEATURES,
            "version": version,
            "reference": build_reference(train, FEATURES),
        },
        out / "model.joblib",
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("saved to", out)
    print(json.dumps(metrics, indent=2))