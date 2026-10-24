import numpy as np

EPS = 1e-4


def build_reference(df, features, bins=10):
    """Training data ke deciles (bin edges) aur har bin ka expected share."""
    ref = {}
    for f in features:
        edges = np.unique(np.quantile(df[f], np.linspace(0, 1, bins + 1)[1:-1]))
        expected = np.bincount(np.digitize(df[f], edges),
                               minlength=len(edges) + 1) / len(df)
        ref[f] = {"edges": edges, "expected": expected}
    return ref


def psi(values, edges, expected):
    """Population Stability Index: naya data training distribution se kitna hata."""
    actual = np.bincount(np.digitize(values, edges),
                         minlength=len(edges) + 1) / len(values)
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value):
    if value < 0.1:
        return "ok"
    if value < 0.25:
        return "warning"
    return "alert"


def ece(y, p, bins=10):
    """Expected Calibration Error."""
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total