import numpy as np
from scipy import sparse

from vcc_model.decode import decode_sparse, delta_to_fc


def test_delta_to_fc_uses_jensen_corrected_formula():
    # A gene with control mean log1p(CP10k)=1.0 (CP10k~1.718) and delta=-0.2
    # should NOT be decoded as a flat exp(delta) count ratio (the original
    # bug this pipeline fixed) — it must use expm1(m+delta)/expm1(m).
    m_ctx = np.array([1.0], dtype=np.float32)
    delta = np.array([-0.2], dtype=np.float32)
    fc = delta_to_fc(m_ctx, delta, target_idx=-1)
    naive_bug = np.exp(-0.2)  # ~0.819 — understates the true drop
    correct = np.expm1(0.8) / np.expm1(1.0)
    assert abs(fc[0] - correct) < 1e-6
    assert abs(fc[0] - naive_bug) > 0.05


def test_delta_to_fc_floors_sparse_genes_to_one():
    m_ctx = np.array([0.001], dtype=np.float32)  # expm1 ~ 0.001 CP10k, below mu_floor
    delta = np.array([-5.0], dtype=np.float32)
    fc = delta_to_fc(m_ctx, delta, target_idx=-1, mu_floor=0.01)
    assert fc[0] == 1.0


def test_delta_to_fc_overrides_target_gene_to_fixed_prior():
    m_ctx = np.array([2.0, 2.0], dtype=np.float32)
    delta = np.array([-10.0, 0.0], dtype=np.float32)  # gene 0 would hit the floor otherwise
    fc = delta_to_fc(m_ctx, delta, target_idx=0, target_fc=0.10, non_target_floor=0.05)
    assert fc[0] == 0.10


def test_delta_to_fc_clips_non_target_fold_change():
    m_ctx = np.array([2.0], dtype=np.float32)
    delta = np.array([10.0], dtype=np.float32)  # huge increase, should be capped
    fc = delta_to_fc(m_ctx, delta, target_idx=-1, cap=20.0)
    assert fc[0] == 20.0


def test_decode_sparse_is_nonnegative_integer_and_deterministic_given_seed():
    rows = sparse.csr_matrix(np.array([[10, 0, 100], [5, 20, 0]], dtype=np.int64))
    fc = np.array([0.1, 2.0, 1.0], dtype=np.float32)
    out_a = decode_sparse(rows, fc, np.random.default_rng(0))
    out_b = decode_sparse(rows, fc, np.random.default_rng(0))
    assert np.array_equal(out_a.toarray(), out_b.toarray())
    assert np.issubdtype(out_a.data.dtype, np.integer)
    assert np.all(out_a.data >= 0)


def test_decode_sparse_thins_down_and_grows_up_in_expectation():
    rng = np.random.default_rng(42)
    rows = sparse.csr_matrix(np.full((2000, 1), 100, dtype=np.int64))
    fc_down = np.array([0.1], dtype=np.float32)
    fc_up = np.array([3.0], dtype=np.float32)
    down = decode_sparse(rows, fc_down, rng).toarray().mean()
    up = decode_sparse(rows, fc_up, rng).toarray().mean()
    assert 5 < down < 15  # ~10 expected
    assert 280 < up < 320  # ~300 expected
