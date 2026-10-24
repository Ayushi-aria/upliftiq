import numpy as np
import pandas as pd
from econml.dr import DRLearner
from econml.metalearners import XLearner
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURES = ["tenure", "activity", "risk_score", "age", "balance"]


def lgbm():
    return LGBMRegressor(n_estimators=200, learning_rate=0.05,
                         num_leaves=15, verbose=-1)


def top_k_gain(score, true_uplift, k=0.2):
    """Top k% users ko score se chuno, unka average TRUE uplift batao."""
    n = int(len(score) * k)
    idx = np.argsort(-score)[:n]
    return true_uplift[idx].mean()


if __name__ == "__main__":
    df = pd.read_csv("data/sim.csv")
    train, test = train_test_split(df, test_size=0.3, random_state=0)

    Xtr, Ytr, Ttr = train[FEATURES], train["Y"], train["T"]
    Xte = test[FEATURES]
    true = test["true_uplift"].values

    # Model 1: X-learner
    xl = XLearner(models=lgbm(), cate_models=lgbm(),
                  propensity_model=LogisticRegression())
    xl.fit(Ytr, Ttr, X=Xtr)

    # Model 2: DR-learner
    dr = DRLearner(model_propensity=LogisticRegression(),
                   model_regression=lgbm(), model_final=lgbm())
    dr.fit(Ytr, Ttr, X=Xtr)

    scores = {
        "X-learner": xl.effect(Xte).ravel(),
        "DR-learner": dr.effect(Xte).ravel(),
        "Risk-score baseline": test["risk_score"].values,
    }

    print("\nTop 20% targeting: avg TRUE uplift of selected users")
    print("Random targeting:", round(true.mean(), 4))
    for name, s in scores.items():
        corr = spearmanr(s, true)[0]
        gain = top_k_gain(s, true)
        print(f"{name:20s} rank-corr={corr:.3f}  top20%-uplift={gain:.4f}")

    out = test.copy()
    out["x_uplift"] = scores["X-learner"]
    out["dr_uplift"] = scores["DR-learner"]
    out.to_csv("data/test_scores.csv", index=False)