"""The structured effect model: predicted_delta = g + lambda(strength)*s + KD + A1 (see README)."""

from __future__ import annotations

from typing import Any

import numpy as np

STRENGTH_BINS = [0, 5, 20, 50, 100, 200, 500, 1000, np.inf]
QUALIFYING_NEIGHBOR_BINS = [(0, 1_000), (1_000, 10_000), (10_000, 50_000)]
CONTROL_DISTANCE = (1_000_000, 5_000_000)


def strength(delta_row: np.ndarray, threshold: float = 0.1) -> int:
    """Count genes whose |delta| exceeds the threshold — the perturbation's strength."""
    return int(np.sum(np.abs(delta_row) > threshold))


def generic_response(H: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each line's mean delta (g_h, g_e) and their average (g_common)."""
    g_h = H.mean(axis=0)
    g_e = E.mean(axis=0)
    return g_h, g_e, (g_h + g_e) / 2.0


def fit_lambda_curve(
    src: np.ndarray,
    dst: np.ndarray,
    g_src: np.ndarray,
    g_dst: np.ndarray,
    names_src: list[str],
    names_dst: list[str],
    bins: list[float] = STRENGTH_BINS,
) -> list[dict[str, float]]:
    """Fit lambda per strength bin: least-squares transfer weight of src's specific effect onto dst's."""
    shared = sorted(set(names_src) & set(names_dst))
    si = np.asarray([names_src.index(t) for t in shared])
    di = np.asarray([names_dst.index(t) for t in shared])
    src_m, dst_m = src[si], dst[di]
    strengths = np.sum(np.abs(src_m) > 0.1, axis=1)
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (strengths >= lo) & (strengths < hi)
        s = src_m[mask] - g_src
        y = dst_m[mask] - g_dst
        denom = float(np.sum(s * s))
        lam = float(np.sum(s * y) / denom) if denom > 0 else 0.0
        curve.append(
            {"lo": float(lo), "hi": float(hi), "n": int(mask.sum()), "lambda": float(np.clip(lam, 0.0, 1.0))}
        )
    return curve


def average_curves(curve_a: list[dict], curve_b: list[dict]) -> list[dict]:
    """Average two directional lambda curves into one symmetric curve."""
    return [
        {"lo": a["lo"], "hi": a["hi"], "lambda_mean": float(np.mean([a["lambda"], b["lambda"]]))}
        for a, b in zip(curve_a, curve_b)
    ]


def lambda_for(value: float, curve: list[dict]) -> float:
    """Look up the lambda for a strength value, falling back to the last bin above range."""
    for row in curve:
        if row["lo"] <= value < row["hi"]:
            return row["lambda_mean"]
    return curve[-1]["lambda_mean"]


def knockdown_delta(
    symbol: str, symbol_cols: dict[str, list[int]], ctrl_means: np.ndarray, target_fc: float = 0.10
) -> float:
    """Return the PIE-space delta implied by the fixed 0.10 knockdown prior for this gene."""
    m = float(np.mean([ctrl_means[0, symbol_cols[symbol][0]], ctrl_means[1, symbol_cols[symbol][0]]]))
    return float(np.log1p(target_fc * np.expm1(m)) - m)


def fit_bin_effects_one_line(
    matrix: np.ndarray,
    idx_map: dict[str, int],
    chrom: dict,
    tss: dict,
    chrom_idx: dict,
    targets: list[str],
    qualifying_bins=QUALIFYING_NEIGHBOR_BINS,
) -> dict[tuple[int, int], float]:
    """Fit one line's median neighbor-vs-matched-control delta per TSS-distance bin."""
    per_bin: dict[tuple[int, int], list[float]] = {b: [] for b in qualifying_bins}
    for t in targets:
        if t not in idx_map or chrom.get(t) is None or chrom[t] not in chrom_idx:
            continue
        delta_row = matrix[idx_map[t]]
        t_tss = tss[t]
        idxs, positions, expr = chrom_idx[chrom[t]]
        dist = np.abs(positions - t_tss)
        not_self = positions != t_tss
        ctrl_mask = (dist >= CONTROL_DISTANCE[0]) & (dist <= CONTROL_DISTANCE[1]) & not_self
        if ctrl_mask.sum() < 5:
            continue
        cand_expr, cand_delta = expr[ctrl_mask], delta_row[idxs[ctrl_mask]]
        order = np.argsort(cand_expr)
        cand_expr_sorted, cand_delta_sorted = cand_expr[order], cand_delta[order]
        for b in qualifying_bins:
            lo, hi = b
            mask = (dist >= lo) & (dist < hi) & not_self
            if not mask.any():
                continue
            nb_expr, nb_delta = expr[mask], delta_row[idxs[mask]]
            pos = np.clip(np.searchsorted(cand_expr_sorted, nb_expr), 0, len(cand_expr_sorted) - 1)
            pos_left = np.clip(pos - 1, 0, len(cand_expr_sorted) - 1)
            use_left = np.abs(cand_expr_sorted[pos_left] - nb_expr) < np.abs(cand_expr_sorted[pos] - nb_expr)
            matched = cand_delta_sorted[np.where(use_left, pos_left, pos)]
            per_bin[b].extend((nb_delta - matched).tolist())
    return {b: (float(np.median(v)) if v else 0.0) for b, v in per_bin.items()}


def fit_a1_bin_effects(
    H: np.ndarray,
    E: np.ndarray,
    h_idx: dict[str, int],
    e_idx: dict[str, int],
    chrom: dict[str, str | None],
    tss: dict[str, int | None],
    chrom_idx: dict,
    targets: list[str],
    qualifying_bins=QUALIFYING_NEIGHBOR_BINS,
) -> dict[tuple[int, int], float]:
    """Fit fit_bin_effects_one_line on both reference lines and average them."""
    eff_h = fit_bin_effects_one_line(H, h_idx, chrom, tss, chrom_idx, targets, qualifying_bins)
    eff_e = fit_bin_effects_one_line(E, e_idx, chrom, tss, chrom_idx, targets, qualifying_bins)
    return {b: float(np.mean([eff_h[b], eff_e[b]])) for b in qualifying_bins}


def neighbor_delta_terms(
    target: str,
    chrom: dict,
    tss: dict,
    chrom_idx: dict,
    bin_effects: dict[tuple[int, int], float],
    self_index: int | None,
) -> list[tuple[int, float]]:
    """Return (gene index, delta) pairs for one target's neighbor genes, excluding self_index."""
    c = chrom.get(target)
    if c is None or c not in chrom_idx:
        return []
    t_tss = tss[target]
    idxs, positions, _ = chrom_idx[c]
    not_self = positions != t_tss
    out = []
    for (lo, hi), eff in bin_effects.items():
        if eff == 0.0:
            continue
        mask = (np.abs(positions - t_tss) >= lo) & (np.abs(positions - t_tss) < hi) & not_self
        for gi in idxs[mask]:
            gi = int(gi)
            if gi != self_index:
                out.append((gi, eff))
    return out


def build_target_delta(
    target: str,
    symbols: list[str],
    g_common: np.ndarray,
    H: np.ndarray,
    E: np.ndarray,
    h_idx: dict[str, int],
    e_idx: dict[str, int],
    curve: list[dict],
    symbol_cols: dict[str, list[int]],
    ctrl_means: np.ndarray,
    k562_map: Any | None,
    chrom: dict,
    tss: dict,
    chrom_idx: dict,
    bin_effects: dict[tuple[int, int], float],
    target_fc: float = 0.10,
) -> tuple[np.ndarray, bool]:
    """Assemble one target's full predicted delta: g + lambda*s + KD + A1."""
    ref = np.full(len(symbols), np.nan, dtype=np.float32)
    s_value = 0
    if target in h_idx:
        hv = H[h_idx[target]]
        s_value = max(s_value, strength(hv))
        ref[:] = hv
    if target in e_idx:
        ev = E[e_idx[target]]
        s_value = max(s_value, strength(ev))
        ref = ev if np.isnan(ref).all() else np.nanmean(np.vstack([ref, ev]), axis=0)
    has_specific = not np.isnan(ref).all()
    if not has_specific and k562_map is not None and k562_map.has_target(target):
        mapped = k562_map.map_target(target, len(symbols))
        ref[:] = mapped
        s_value = strength(mapped)
        has_specific = not np.isnan(ref).all()

    delta = g_common.copy()
    self_index = symbols.index(target) if target in symbols else None
    if has_specific:
        s = np.nan_to_num(ref - g_common, nan=0.0)
        lam = lambda_for(s_value, curve)
        delta += lam * s
    if self_index is not None:
        delta[self_index] += knockdown_delta(target, symbol_cols, ctrl_means, target_fc)
    for gi, eff in neighbor_delta_terms(target, chrom, tss, chrom_idx, bin_effects, self_index):
        delta[gi] += eff
    return delta, has_specific


def build_all_predictions(final_targets: list[str], challenge_genes: list[str], symbols: list[str], **kwargs):
    """Assemble the full (n_targets x n_challenge_genes) delta matrix."""
    gidx = {g: i for i, g in enumerate(challenge_genes)}
    out = np.zeros((len(final_targets), len(challenge_genes)), dtype=np.float32)
    has_specific = []
    for ti, target in enumerate(final_targets):
        delta, specific = build_target_delta(target, symbols, **kwargs)
        for j, sym in enumerate(symbols):
            out[ti, gidx[sym]] = delta[j]
        has_specific.append(specific)
    return out, has_specific
