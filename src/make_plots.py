from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.metalearners import XLearner
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.train import FEATURES, lgbm

OUT = Path("docs")


def benefit_curve(score, true, max_benefit):
    """Score ke order mein users chuno, cumulative benefit (% of max) nikalo."""
    order = np.argsort(-score)
    cum = np.cumsum(true[order]) / max_benefit * 100
    frac = np.arange(1, len(true) + 1) / len(true) * 100
    return frac, cum


def reliability(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    xs, ys = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 30:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return xs, ys


def plot_uplift_vs_budget(df):
    train, test = train_test_split(df, test_size=0.3, random_state=0)
    xl = XLearner(models=lgbm(), cate_models=lgbm(),
                  propensity_model=LogisticRegression())
    xl.fit(train["Y"], train["T"], X=train[FEATURES])

    true = test["true_uplift"].values
    max_benefit = true[true > 0].sum()
    curves = {
        "X-learner (uplift)": xl.effect(test[FEATURES]).ravel(),
        "Risk-score targeting": test["risk_score"].values,
        "Oracle (true uplift)": true,
    }

    plt.figure(figsize=(7, 5))
    for name, s in curves.items():
        x, y = benefit_curve(s, true, max_benefit)
        plt.plot(x, y, label=name)
    plt.plot([0, 100], [0, true.sum() / max_benefit * 100],
             "--", color="gray", label="Random targeting")
    plt.axhline(0, color="black", lw=0.5)
    plt.xlabel("% of users targeted (budget)")
    plt.ylabel("% of max achievable benefit")
    plt.title("Uplift vs budget (simulated data)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "uplift_vs_budget.png", dpi=150)
    plt.close()

    k = int(len(true) * 0.2)
    for name in ("X-learner (uplift)", "Risk-score targeting"):
        top = np.argsort(-curves[name])[:k]
        print(f"Top 20% by {name}: captured "
              f"{true[top].sum() / max_benefit * 100:.1f}% of max achievable benefit")


def plot_reliability(df):
    train, rest = train_test_split(df, test_size=0.5, random_state=0)
    calib, test = train_test_split(rest, test_size=0.6, random_state=0)

    tr = train[train["T"] == 0]
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                         num_leaves=31, verbose=-1)
    clf.fit(tr[FEATURES], tr["Y"])

    c = calib[calib["T"] == 0]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(clf.predict_proba(c[FEATURES])[:, 1], c["Y"])

    te = test[test["T"] == 0]
    raw = clf.predict_proba(te[FEATURES])[:, 1]
    cal = iso.predict(raw)

    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for name, p in (("Raw", raw), ("Isotonic-calibrated", cal)):
        x, y = reliability(te["Y"].values, p)
        plt.plot(x, y, marker="o", label=name)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed outcome rate")
    plt.title("Reliability diagram (control arm)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "reliability.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv("data/sim.csv")
    plot_uplift_vs_budget(data)
    plot_reliability(data)
    print("saved plots to", OUT)