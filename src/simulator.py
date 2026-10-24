import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def simulate(n=50_000, seed=42, treat_prob=0.5):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "tenure": rng.exponential(2.0, n),
        "activity": rng.normal(0, 1, n),
        "risk_score": rng.normal(0, 1, n),
        "age": rng.integers(18, 70, n),
        "balance": rng.lognormal(0, 1, n),
    })

    # Baseline: treatment ke bina outcome ki probability (risk_score se driven)
    logit0 = -1.0 + 0.8 * df.risk_score + 0.4 * df.activity - 0.1 * df.tenure

    # Treatment effect: activity se badhta hai, high-risk users pe ulta (backfire)
    effect = 0.9 * df.activity - 1.2 * (df.risk_score > 1.0)

    p0 = sigmoid(logit0)             # P(Y=1 | no treatment)
    p1 = sigmoid(logit0 + effect)    # P(Y=1 | treatment)

    df["true_uplift"] = p1 - p0      # ground truth, models ko ye nahi dikhega
    df["T"] = rng.binomial(1, treat_prob, n)   # random assignment (A/B test)
    df["Y"] = rng.binomial(1, np.where(df["T"] == 1, p1, p0))
    return df


if __name__ == "__main__":
    df = simulate()
    df.to_csv("data/sim.csv", index=False)
    print("rows:", len(df))
    print("true ATE:", round(df.true_uplift.mean(), 4))
    print("sleeping dogs (uplift < 0):", round((df.true_uplift < 0).mean(), 3))
    print("corr(risk_score, uplift):", round(df.risk_score.corr(df.true_uplift), 3))