import numpy as np
import pandas as pd

from src.monitor import build_reference, psi, psi_status


def _ref():
    rng = np.random.default_rng(0)
    ref = build_reference(pd.DataFrame({"x": rng.normal(size=20000)}), ["x"])
    return rng, ref["x"]["edges"], ref["x"]["expected"]


def test_psi_is_small_without_drift():
    rng, edges, expected = _ref()
    assert psi(rng.normal(size=5000), edges, expected) < 0.1


def test_psi_is_large_when_distribution_shifts():
    rng, edges, expected = _ref()
    assert psi(rng.normal(loc=1.5, size=5000), edges, expected) > 0.25


def test_status_thresholds():
    assert psi_status(0.05) == "ok"
    assert psi_status(0.15) == "warning"
    assert psi_status(0.40) == "alert"