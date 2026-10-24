import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.train import FEATURES, top_k_gain


def ece(y, p, bins=10):
    """Expected Calibration Error: predicted prob vs actual rate ka average gap."""
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


if __name__ == "__main__":
    df = pd.read_csv("data/sim.csv")
    train, rest = train_test_split(df, test_size=0.5, random_state=0)
    calib, test = train_test_split(rest, test_size=0.6, random_state=0)

    raw, cal = {}, {}
    for t in (0, 1):
        tr = train[train["T"] == t]
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             num_leaves=31, verbose=-1)
        clf.fit(tr[FEATURES], tr["Y"])

        # isotonic calibration, alag calibration set pe fit hota hai
        c = calib[calib["T"] == t]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(c[FEATURES])[:, 1], c["Y"])

        raw[t] = clf.predict_proba(test[FEATURES])[:, 1]
        cal[t] = iso.predict(raw[t])

        m = (test["T"] == t).values
        y = test["Y"].values[m]
        print(f"Arm T={t}: ECE raw={ece(y, raw[t][m]):.4f}  "
              f"calibrated={ece(y, cal[t][m]):.4f}")

    true = test["true_uplift"].values
    up_raw = raw[1] - raw[0]
    up_cal = cal[1] - cal[0]
    print("\nUplift (p1 - p0) quality:")
    for name, s in [("raw", up_raw), ("calibrated", up_cal)]:
        print(f"{name:11s} rank-corr={spearmanr(s, true)[0]:.3f}  "
              f"top20%-uplift={top_k_gain(s, true):.4f}")

    out = test.copy()
    out["p0_cal"], out["p1_cal"] = cal[0], cal[1]
    out["uplift_cal"] = up_cal
    out.to_csv("data/calibrated_scores.csv", index=False)