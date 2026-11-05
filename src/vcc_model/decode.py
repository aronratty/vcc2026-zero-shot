"""Delta (PIE log1p space) -> count-space fold change -> integer counts.

PIE's delta is a difference in log1p(CP10k)-normalized space, not a count
fold change — `exp(delta)` as a count ratio understates every change for
low/medium-expression genes (Jensen's inequality: `expm1(mean(log1p(x)))`
is below the mean count-scale expression in general). The correct
count-space fold change, given a context's own control mean `m_ctx`, is:

    fc = expm1(m_ctx + delta) / expm1(m_ctx)

Non-target genes below a CP10k floor keep fc=1 (too sparse to trust);
other non-target fold changes are clipped to [floor, cap]. Every target
gene uses the fixed challenge prior `target_fc` exactly (0.10 by default;
Arc's disclosed >=80% guide-selection rule, not fit locally — see README
"What's learned vs fixed").
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import sparse


def context_mean_log1p(control_X, ntc: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Mean log1p(CP10k) over one context's own non-targeting control cells."""
    total = np.zeros(control_X.shape[1], dtype=np.float64)
    for start in range(0, len(ntc), chunk_size):
        rows = control_X[ntc[start : start + chunk_size]].tocsr(copy=True)
        totals = np.asarray(rows.sum(axis=1)).ravel().astype(np.float64)
        rows = rows.multiply((10000.0 / np.maximum(totals, 1.0))[:, None]).tocsr()
        rows.data = np.log1p(rows.data)
        total += np.asarray(rows.sum(axis=0)).ravel()
    return (total / max(len(ntc), 1)).astype(np.float32)


def delta_to_fc(
    m_ctx: np.ndarray,
    delta: np.ndarray,
    target_idx: int,
    target_fc: float = 0.10,
    mu_floor: float = 0.01,
    non_target_floor: float = 0.05,
    cap: float = 20.0,
) -> np.ndarray:
    mu = np.expm1(m_ctx.astype(np.float64))
    fc = np.ones(delta.shape[0], dtype=np.float64)
    usable = mu >= mu_floor
    fc[usable] = np.expm1(m_ctx[usable].astype(np.float64) + delta[usable].astype(np.float64)) / mu[usable]
    fc[usable] = np.clip(fc[usable], non_target_floor, cap)
    if target_idx >= 0:
        fc[target_idx] = target_fc
    return fc.astype(np.float32)


def decode_sparse(
    control_rows: sparse.csr_matrix, fc: np.ndarray, rng: np.random.Generator
) -> sparse.csr_matrix:
    """fc<1: Binomial(count, fc) thinning. fc>1: count + Poisson(count*(fc-1))."""
    out = control_rows.tocsr(copy=True)
    if not np.issubdtype(out.data.dtype, np.integer):
        out.data = np.rint(out.data).astype(np.int64)
    row_fc = fc[out.indices]
    below = row_fc < 1.0
    values = out.data.astype(np.int64, copy=False)
    if below.any():
        values[below] = rng.binomial(values[below], row_fc[below])
    if (~below).any():
        values[~below] = values[~below] + rng.poisson(values[~below] * (row_fc[~below] - 1.0))
    out.data = values.astype(np.int32, copy=False)
    out.eliminate_zeros()
    out.sort_indices()
    return out


def decode_context(
    control_h5ad_path,
    context: str,
    genes: list[str],
    targets: list[str],
    deltas: np.ndarray,
    seed: int,
    cells_per_pert: int = 400,
    target_fc: float = 0.10,
    mu_floor: float = 0.01,
    verbose: bool = True,
):
    """Decode every target's predicted cells for one context. Yields
    (target, decoded_csr_block, obs_rows) per target, streamed to keep
    memory bounded for 300 targets x 18,533 genes."""
    import anndata as ad

    controls = ad.read_h5ad(control_h5ad_path, backed="r")
    ntc = np.flatnonzero(controls.obs["target_gene"].astype(str).to_numpy() == "non-targeting")
    if len(ntc) == 0:
        raise ValueError(f"no non-targeting cells in {control_h5ad_path}")
    rng = np.random.default_rng(seed)
    m_ctx = context_mean_log1p(controls.X, ntc)
    if verbose:
        print(f"{context}: m_ctx range {m_ctx.min():.4g}..{m_ctx.max():.4g}", file=sys.stderr, flush=True)
    for target_idx, target in enumerate(targets):
        chosen = rng.choice(ntc, size=cells_per_pert, replace=True)
        rows = controls.X[chosen].tocsr(copy=True)
        target_gene_idx = genes.index(target) if target in genes else -1
        fc = delta_to_fc(m_ctx, deltas[target_idx], target_gene_idx, target_fc=target_fc, mu_floor=mu_floor)
        decoded = decode_sparse(rows, fc, rng)
        obs_rows = [{"target_gene": target, "context": context} for _ in range(decoded.shape[0])]
        yield target, decoded, obs_rows
        if verbose and (target_idx + 1) % 25 == 0:
            print(f"{context}: {target_idx + 1}/{len(targets)} targets", file=sys.stderr, flush=True)
    controls.file.close()
