import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.train import FEATURES

ALPHA = 0.1  # 90% coverage


def conformal_threshold(p1, y, alpha):
    # nonconformity score = 1 - (true class ko di gayi probability)
    s = np.where(y == 1, 1 - p1, p1)
    n = len(s)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(s, level, method="higher")


def prediction_sets(p1, qhat):
    has0 = p1 <= qhat
    has1 = (1 - p1) <= qhat
    return has0, has1


if __name__ == "__main__":
    df = pd.read_csv("data/sim.csv")
    train, rest = train_test_split(df, test_size=0.6, random_state=0)
    iso_set, rest = train_test_split(rest, test_size=2 / 3, random_state=0)
    conf, test = train_test_split(rest, test_size=0.5, random_state=0)

    for t in (0, 1):
        tr, iso_t = train[train["T"] == t], iso_set[iso_set["T"] == t]
        cf, te = conf[conf["T"] == t], test[test["T"] == t]

        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             num_leaves=31, verbose=-1)
        clf.fit(tr[FEATURES], tr["Y"])

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(iso_t[FEATURES])[:, 1], iso_t["Y"])

        def prob(d, clf=clf, iso=iso):
            return iso.predict(clf.predict_proba(d[FEATURES])[:, 1])

        qhat = conformal_threshold(prob(cf), cf["Y"].values, ALPHA)

        has0, has1 = prediction_sets(prob(te), qhat)
        y = te["Y"].values
        covered = np.where(y == 1, has1, has0)
        size = has0.astype(int) + has1.astype(int)

        print(f"Arm T={t}: qhat={qhat:.3f}  coverage={covered.mean():.3f}  "
              f"avg set size={size.mean():.2f}  confident (size 1)={np.mean(size == 1):.1%}")