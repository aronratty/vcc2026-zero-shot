"""Submission integrity checks: shape, labels, counts, and target-gene ratio.

These are independent sanity checks on a decoded prediction file, not a
biological validation — there is no local D/E/F ground truth to validate
against (see METHODS.md "Local-test limitations").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


def label_integrity(
    h5ad_path: str | Path, expected_contexts: set[str], cells_per_pert: int = 400
) -> dict[str, Any]:
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs
    counts = obs.groupby(["context", "target_gene"], observed=True).size()
    report = {
        "shape": list(a.shape),
        "contexts": sorted(set(obs["context"].astype(str))),
        "contexts_match_expected": set(obs["context"].astype(str)) == set(expected_contexts),
        "target_context_groups": int(len(counts)),
        "groups_with_wrong_cell_count": int((counts != cells_per_pert).sum()),
        "non_targeting_labels_present": sorted(set(obs["target_gene"].astype(str)) & {"non-targeting"}),
    }
    a.file.close()
    return report


def nonneg_integer_counts_sampled(
    h5ad_path: str | Path, n_blocks: int = 72, block_size: int = 5000
) -> dict[str, Any]:
    a = ad.read_h5ad(h5ad_path, backed="r")
    scan_starts = np.linspace(0, a.n_obs - block_size, n_blocks, dtype=int)
    min_v, max_v, non_integer, zero_total, nnz = np.inf, -np.inf, 0, 0, 0
    for start in scan_starts:
        block = a.X[start : start + block_size].tocsr()
        if block.nnz:
            min_v = min(min_v, float(block.data.min()))
            max_v = max(max_v, float(block.data.max()))
            non_integer += int(np.count_nonzero(block.data != np.floor(block.data)))
            nnz += int(block.nnz)
        zero_total += int(np.count_nonzero(np.asarray(block.sum(axis=1)).ravel() <= 0))
    a.file.close()
    return {
        "sampled_row_blocks": len(scan_starts),
        "min_nonzero_sampled": min_v,
        "max_nonzero_sampled": max_v,
        "non_integer_nonzeros_sampled": non_integer,
        "zero_total_cells_sampled": zero_total,
        "nnz_sampled": nnz,
    }


def target_gene_ratio_report(
    h5ad_path: str | Path,
    control_paths: dict[str, Path],
    targets: list[str],
    genes: list[str],
    cells_per_pert: int = 400,
) -> dict[str, Any]:
    """Predicted/control mean-count ratio for each target's own gene, per
    context — should land near the fixed knockdown prior (~0.10)."""
    out = ad.read_h5ad(h5ad_path, backed="r")
    gidx = {g: i for i, g in enumerate(genes)}
    contexts = sorted(control_paths)
    rows = []
    for cidx, context in enumerate(contexts):
        ctrl = ad.read_h5ad(control_paths[context], backed="r")
        ctrl_mem = ctrl.X[:].tocsr()
        ntc = np.flatnonzero(ctrl.obs["target_gene"].astype(str).to_numpy() == "non-targeting")
        for tidx, target in enumerate(targets):
            j = gidx[target]
            start = cidx * len(targets) * cells_per_pert + tidx * cells_per_pert
            pred_vals = out.X[start : start + cells_per_pert].tocsr()[:, j].toarray().ravel()
            ctrl_mean = float(ctrl_mem[ntc, j].toarray().ravel().mean())
            rows.append(
                {
                    "context": context,
                    "target_gene": target,
                    "pred_mean": float(pred_vals.mean()),
                    "control_mean": ctrl_mean,
                    "ratio": float(pred_vals.mean() / ctrl_mean) if ctrl_mean > 0 else None,
                }
            )
        ctrl.file.close()
    out.file.close()
    df = pd.DataFrame(rows)
    summary = {}
    for context in contexts:
        x = df[df.context == context]
        summary[context] = {
            "median_ratio": float(x.ratio.dropna().median()),
            "mean_ratio": float(x.ratio.dropna().mean()),
            "n_nonzero_control": int(x.ratio.notna().sum()),
        }
    return {"per_target": df, "summary": summary}
