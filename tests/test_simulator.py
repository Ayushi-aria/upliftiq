from src.simulator import simulate


def test_shapes_and_ranges():
    df = simulate(n=2000, seed=1)
    assert len(df) == 2000
    assert set(df["T"].unique()) <= {0, 1}
    assert set(df["Y"].unique()) <= {0, 1}
    assert df["true_uplift"].between(-1, 1).all()


def test_reproducible_with_same_seed():
    a = simulate(n=500, seed=7)
    b = simulate(n=500, seed=7)
    assert a.equals(b)


def test_sleeping_dogs_exist_among_high_risk():
    df = simulate(n=5000, seed=3)
    high_risk = df[df["risk_score"] > 1.5]
    assert high_risk["true_uplift"].mean() < 0