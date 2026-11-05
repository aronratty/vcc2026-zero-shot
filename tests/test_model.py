import numpy as np

from vcc_model import model


def test_strength_counts_genes_above_threshold():
    row = np.array([0.0, 0.2, -0.3, 0.05, 1.0])
    assert model.strength(row, threshold=0.1) == 3


def test_lambda_for_picks_correct_bin_and_falls_back_to_last():
    curve = [{"lo": 0, "hi": 5, "lambda_mean": 0.1}, {"lo": 5, "hi": 20, "lambda_mean": 0.2}]
    assert model.lambda_for(2, curve) == 0.1
    assert model.lambda_for(5, curve) == 0.2
    assert model.lambda_for(999, curve) == 0.2  # beyond the last bin's hi -> last bin


def test_knockdown_delta_is_negative_for_a_fc_below_one():
    symbol_cols = {"TP53": [0]}
    ctrl_means = np.array([[2.0], [2.0]], dtype=np.float32)  # log1p control mean of 2.0 in both lines
    delta = model.knockdown_delta("TP53", symbol_cols, ctrl_means, target_fc=0.10)
    assert delta < 0
    # recovering the fold change should give back ~0.10
    fc = np.expm1(2.0 + delta) / np.expm1(2.0)
    assert abs(fc - 0.10) < 1e-5


def test_neighbor_delta_terms_never_touches_self_index():
    symbols = ["A", "B", "C", "D"]
    chrom = {"A": "chr1", "B": "chr1", "C": "chr1", "D": "chr1"}
    tss = {"A": 1000, "B": 1500, "C": 50000, "D": 2_000_000}
    idxs = np.array([0, 1, 2, 3])
    positions = np.array([1000, 1500, 50000, 2_000_000])
    ctrl_expr = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    chrom_idx = {"chr1": (idxs, positions, ctrl_expr)}
    bin_effects = {(0, 1_000): -0.1, (1_000, 10_000): -0.05, (10_000, 50_000): -0.01}

    self_index = symbols.index("A")
    terms = model.neighbor_delta_terms("A", chrom, tss, chrom_idx, bin_effects, self_index)
    touched = {gi for gi, _ in terms}
    assert self_index not in touched
    # B is 500bp away (1-10kb bin), should be touched
    assert symbols.index("B") in touched


def test_build_target_delta_uses_knockdown_prior_when_no_reference_available():
    symbols = ["TP53", "GENE2"]
    g_common = np.zeros(2, dtype=np.float32)
    H = np.zeros((0, 2), dtype=np.float32)
    E = np.zeros((0, 2), dtype=np.float32)
    symbol_cols = {"TP53": [0], "GENE2": [1]}
    ctrl_means = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    delta, has_specific = model.build_target_delta(
        "TP53",
        symbols,
        g_common,
        H,
        E,
        {},
        {},
        curve=[{"lo": 0, "hi": np.inf, "lambda_mean": 0.5}],
        symbol_cols=symbol_cols,
        ctrl_means=ctrl_means,
        k562_map=None,
        chrom={},
        tss={},
        chrom_idx={},
        bin_effects={},
    )
    assert has_specific is False
    assert delta[0] < 0  # the knockdown prior landed on TP53's own column
    assert delta[1] == 0.0  # no effect anywhere else
