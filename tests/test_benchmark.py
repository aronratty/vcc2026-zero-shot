import numpy as np

from vcc_model import benchmark


def test_top200_scores_perfect_prediction():
    true = np.random.default_rng(0).normal(size=500)
    scores = benchmark.top200_scores(true.copy(), true)
    assert scores["mae_top200"] == 0.0
    assert scores["sign_top200"] == 1.0
    assert scores["overlap_top200"] == 1.0


def test_top200_scores_sign_flip_prediction():
    true = np.zeros(500)
    true[:200] = np.linspace(1, 10, 200)  # true top-200 are these
    pred = -true
    scores = benchmark.top200_scores(pred, true)
    assert scores["sign_top200"] == 0.0
    assert scores["overlap_top200"] == 1.0  # same genes ranked top-200 by magnitude


def test_bootstrap_ci_tight_around_mean_for_low_variance_data():
    values = np.full(1000, -0.01)
    lo, hi = benchmark.bootstrap_ci(values, n_boot=500)
    assert lo < -0.01 < hi or abs(lo - hi) < 1e-9
    assert -0.02 < lo <= hi < 0.0


def test_bootstrap_ci_excludes_zero_for_a_consistent_nonzero_effect():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=-0.05, scale=0.01, size=2000)
    lo, hi = benchmark.bootstrap_ci(values, n_boot=1000)
    assert hi < 0


def test_even_odd_split_is_deterministic_and_covers_all_names():
    names = ["GENE3", "GENE1", "GENE2", "GENE4", "GENE5"]
    evens, odds = benchmark.even_odd_split(names)
    assert sorted(evens + odds) == sorted(names)
    assert set(evens).isdisjoint(odds)
    # same input always gives the same split (leakage-safety depends on this)
    evens2, odds2 = benchmark.even_odd_split(names)
    assert evens == evens2 and odds == odds2
