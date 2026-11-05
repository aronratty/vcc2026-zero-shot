"""Derive reference-line deltas, the K562 rank->magnitude map, and a
TSS/chromosome index, from the raw caches loaded by data.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def align_gene_axis(
    xatlas: dict[str, Any], challenge_genes: list[str]
) -> tuple[list[str], dict[str, list[int]]]:
    """Intersect the PIE gene axis with the challenge gene list by symbol.

    Returns (symbols, symbol_cols): symbols is the sorted intersection (the
    axis every reference-line matrix uses below); symbol_cols maps each
    symbol to its column index(es) in the raw PIE gene axis (for ctrl_means
    lookups, which are indexed on the raw axis, not `symbols`).
    """
    gidx = {g: i for i, g in enumerate(challenge_genes)}
    gmeta = xatlas["gene_metadata"]
    ens_to_symbol = dict(zip(gmeta.ensembl_id, gmeta.gene_name))
    known = set(gmeta.gene_name)
    raw_symbols = [x if x in known else ens_to_symbol.get(x) for x in xatlas["meta"]["genes"]]
    symbol_cols: dict[str, list[int]] = {}
    for j, s in enumerate(raw_symbols):
        if s in gidx:
            symbol_cols.setdefault(s, []).append(j)
    return sorted(symbol_cols), symbol_cols


def load_reference_line(
    xatlas: dict[str, Any], context_name: str, cols: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Per-perturbation pseudobulk delta matrix for one reference cell line
    (e.g. "hct116" or "hek293t"), restricted to the aligned `cols` axis.
    """
    meta = xatlas["meta"]
    inv = {v: k for k, v in meta["pert_to_id"].items()}
    cid = meta["context_to_id"][context_name]
    ctx, pert = xatlas["ctx_ids"], xatlas["pert_ids"]
    rows: dict[str, int] = {}
    for r in np.flatnonzero(ctx == cid):
        rows.setdefault(inv[int(pert[r])], int(r))
    names = sorted(rows)
    delta = xatlas["delta_p"]
    out = np.empty((len(names), len(cols)), dtype=np.float32)
    for start in range(0, len(names), 256):
        block_rows = [rows[n] for n in names[start : start + 256]]
        out[start : start + len(block_rows)] = np.asarray(delta[block_rows][:, cols], dtype=np.float32)
    return names, out


def control_expression(
    xatlas: dict[str, Any], symbols: list[str], symbol_cols: dict[str, list[int]]
) -> np.ndarray:
    """Mean log1p(CP10k) control expression per symbol, averaged over the
    two reference lines (ctrl_means rows 0 and 1 in the PIE cache)."""
    ctrl = xatlas["ctrl_means"]
    return np.asarray(
        [float(np.mean([ctrl[0, symbol_cols[s][0]], ctrl[1, symbol_cols[s][0]]])) for s in symbols],
        dtype=np.float32,
    )


class K562QuantileMap:
    """Signed-rank quantile map from K562 CD-coefficient magnitude to the
    HCT116 compact-delta magnitude scale, fit on shared targets.

    K562's processed values are CD coefficients (a characteristic-direction
    vector vs. controls), not log1p deltas — their genome-wide magnitude
    scale does not transfer directly (METHODS.md), so only the *rank* of a
    K562 target's effect per gene is used; the magnitude is remapped onto
    HCT116's own delta distribution on the genes the two share.
    """

    def __init__(
        self,
        k562: dict[str, Any],
        challenge_genes: list[str],
        symbols: list[str],
        H: np.ndarray,
        h_idx: dict[str, int],
    ):
        k_gene_names = [challenge_genes[i] for i in k562["challenge_gene_idx"]]
        k_map = {g: i for i, g in enumerate(k_gene_names)}
        self.q_symbols = [s for s in symbols if s in k_map]
        self.q_h_cols = np.asarray([symbols.index(s) for s in self.q_symbols], dtype=int)
        self.q_k_cols = np.asarray([k_map[s] for s in self.q_symbols], dtype=int)
        self.k_row = {t: i for i, t in enumerate(k562["target_gene"])}
        self.k_values = k562["effect"]

        shared = sorted(set(self.k_row) & set(h_idx))
        h_shared = np.asarray([H[h_idx[t], self.q_h_cols] for t in shared], dtype=np.float32)
        self.rank_magnitude = np.median(np.sort(np.abs(h_shared), axis=1), axis=0)
        self.rank_axis = np.linspace(0.0, 1.0, len(self.rank_magnitude))

    def has_target(self, target: str) -> bool:
        return target in self.k_row

    def map_target(self, target: str, n_symbols: int) -> np.ndarray:
        """Signed, rank-remapped K562 delta for `target`, on the `symbols` axis."""
        raw = self.k_values[self.k_row[target], self.q_k_cols].astype(np.float32)
        order = np.argsort(np.abs(raw))
        q = np.linspace(0.0, 1.0, len(raw))
        mapped_sorted = np.interp(q, self.rank_axis, self.rank_magnitude)
        mapped = np.zeros(len(raw), dtype=np.float32)
        mapped[order] = np.sign(raw[order]) * mapped_sorted
        out = np.zeros(n_symbols, dtype=np.float32)
        out[self.q_h_cols] = mapped
        return out


def build_chrom_index(symbols: list[str], tss_table: dict[str, tuple[str, int]], ctrl_expr: np.ndarray):
    """Per-chromosome (gene-index, tss-position, ctrl-expression) arrays for
    the A1 genomic-neighbor lookups in model.py / benchmark.py."""
    by_chrom: dict[str, list[int]] = {}
    chrom: dict[str, str | None] = {}
    tss: dict[str, int | None] = {}
    for i, s in enumerate(symbols):
        rec = tss_table.get(s)
        if rec is None:
            chrom[s], tss[s] = None, None
            continue
        c, t = rec
        chrom[s], tss[s] = c, t
        by_chrom.setdefault(c, []).append(i)
    index = {}
    for c, idxs in by_chrom.items():
        idxs_arr = np.asarray(idxs)
        positions = np.asarray([tss[symbols[i]] for i in idxs_arr], dtype=np.int64)
        index[c] = (idxs_arr, positions, ctrl_expr[idxs_arr])
    return chrom, tss, index
