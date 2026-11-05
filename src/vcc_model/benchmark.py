"""HCT116<->HEK293T leave-one-line-out (LOLO) benchmark that gates A1/A2 (see RESULTS.md)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from . import model

ALL_NEIGHBOR_BINS = [(0, 1_000), (1_000, 10_000), (10_000, 50_000), (50_000, 200_000)]


def top200_scores(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Score a prediction against truth on the true top-200 DE genes: MAE, sign, overlap."""
    top = np.argsort(np.abs(true))[-200:]
    top_p = set(np.argsort(np.abs(pred))[-200:])
    return {
        "mae_top200": float(np.mean(np.abs(pred[top] - true[top]))),
        "sign_top200": float(np.mean(np.sign(pred[top]) == np.sign(true[top]))),
        "overlap_top200": float(len(top_p & set(top)) / 200.0),
    }


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 2026) -> tuple[float, float]:
    """Return a bootstrap 95% CI of the mean of `values`."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    n = len(values)
    boots = np.asarray([np.mean(values[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def _aggregate_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean each score metric across a list of per-target score dicts."""
    return {k: float(np.mean([r[k] for r in rows])) for k in ["mae_top200", "sign_top200", "overlap_top200"]}


def even_odd_split(names: list[str]) -> tuple[list[str], list[str]]:
    """Split sorted names into alternating halves to fit on one and score on the other."""
    ordered = sorted(names)
    return ordered[0::2], ordered[1::2]


def strength_transfer_spearman(H, E, h_idx, e_idx) -> tuple[float, float, int]:
    """Spearman correlation of perturbation strength between the two lines — A2's go/no-go gate."""
    shared = sorted(set(h_idx) & set(e_idx))
    sh = np.array([model.strength(H[h_idx[t]]) for t in shared])
    se = np.array([model.strength(E[e_idx[t]]) for t in shared])
    rho, p = spearmanr(sh, se)
    return float(rho), float(p), len(shared)


def neighbor_bin_gate_stats(
    H, E, h_idx, e_idx, chrom, tss, chrom_idx, shared_with_tss, bins=ALL_NEIGHBOR_BINS
) -> list[dict[str, Any]]:
    """Report n, median delta, frac negative, and a bootstrap CI per TSS-distance bin and line."""

    def collect(matrix, idx_map):
        """Collect paired (neighbor, matched-control) deltas per bin for one line."""
        res = {b: {"neighbor": [], "control": []} for b in bins}
        for t in shared_with_tss:
            if t not in idx_map or chrom.get(t) is None or chrom[t] not in chrom_idx:
                continue
            delta_row = matrix[idx_map[t]]
            t_tss = tss[t]
            idxs, positions, expr = chrom_idx[chrom[t]]
            dist = np.abs(positions - t_tss)
            not_self = positions != t_tss
            ctrl_mask = (dist >= model.CONTROL_DISTANCE[0]) & (dist <= model.CONTROL_DISTANCE[1]) & not_self
            if ctrl_mask.sum() < 5:
                continue
            cand_expr, cand_delta = expr[ctrl_mask], delta_row[idxs[ctrl_mask]]
            order = np.argsort(cand_expr)
            cand_expr_sorted, cand_delta_sorted = cand_expr[order], cand_delta[order]
            for b in bins:
                lo, hi = b
                mask = (dist >= lo) & (dist < hi) & not_self
                if not mask.any():
                    continue
                nb_expr, nb_delta = expr[mask], delta_row[idxs[mask]]
                pos = np.clip(np.searchsorted(cand_expr_sorted, nb_expr), 0, len(cand_expr_sorted) - 1)
                pos_left = np.clip(pos - 1, 0, len(cand_expr_sorted) - 1)
                use_left = np.abs(cand_expr_sorted[pos_left] - nb_expr) < np.abs(
                    cand_expr_sorted[pos] - nb_expr
                )
                matched = cand_delta_sorted[np.where(use_left, pos_left, pos)]
                res[b]["neighbor"].extend(nb_delta.tolist())
                res[b]["control"].extend(matched.tolist())
        return res

    rows = []
    for label, matrix, idx_map in [("HCT116", H, h_idx), ("HEK293T", E, e_idx)]:
        res = collect(matrix, idx_map)
        for b in bins:
            nb, ct = np.asarray(res[b]["neighbor"]), np.asarray(res[b]["control"])
            diff = nb - ct
            lo_ci, hi_ci = bootstrap_ci(diff) if len(diff) else (float("nan"), float("nan"))
            rows.append(
                {
                    "line": label,
                    "bin": f"{b[0]}-{b[1]}",
                    "n": int(len(nb)),
                    "median_delta": float(np.median(nb)) if len(nb) else float("nan"),
                    "frac_negative": float(np.mean(nb < 0)) if len(nb) else float("nan"),
                    "diff_mean": float(np.mean(diff)) if len(diff) else float("nan"),
                    "diff_ci_lo": lo_ci,
                    "diff_ci_hi": hi_ci,
                }
            )
    return rows


def _baseline_and_a1_prediction(
    src_v, true_len, g_src, g_dst, lam, kd_j, kd, bin_effects, target, chrom, tss, chrom_idx
):
    """Build the g+lambda*s+KD baseline prediction and its +A1 variant for one target."""
    s = src_v - g_src
    pred = g_dst + lam * s
    if kd_j is not None:
        pred = pred.copy()
        pred[kd_j] = g_dst[kd_j] + lam * s[kd_j] + kd
    pred_a1 = pred.copy()
    for gi, eff in model.neighbor_delta_terms(target, chrom, tss, chrom_idx, bin_effects, kd_j):
        pred_a1[gi] += eff
    return pred, pred_a1


def score_a1_direction(
    src_m,
    dst_m,
    src_idx,
    dst_idx,
    g_src,
    g_dst,
    curve,
    fit_targets,
    score_targets,
    chrom,
    tss,
    chrom_idx,
    ctrl_means,
    symbols,
    symbol_cols,
) -> dict[str, Any]:
    """Fit A1 on fit_targets, score baseline vs +A1 on score_targets, one transfer direction."""
    bin_effects = model.fit_bin_effects_one_line(src_m, src_idx, chrom, tss, chrom_idx, fit_targets)
    base_scores, a1_scores = [], []
    for t in score_targets:
        if t not in src_idx or t not in dst_idx:
            continue
        src_v, true = src_m[src_idx[t]], dst_m[dst_idx[t]]
        lam = model.lambda_for(model.strength(src_v), curve)
        kd_j, kd = None, 0.0
        if t in symbols:
            kd_j = symbols.index(t)
            kd = model.knockdown_delta(t, symbol_cols, ctrl_means)
        pred, pred_a1 = _baseline_and_a1_prediction(
            src_v, len(true), g_src, g_dst, lam, kd_j, kd, bin_effects, t, chrom, tss, chrom_idx
        )
        base_scores.append(top200_scores(pred, true))
        a1_scores.append(top200_scores(pred_a1, true))
    return {
        "bin_effects": {f"{b[0]}-{b[1]}": v for b, v in bin_effects.items()},
        "baseline": _aggregate_scores(base_scores),
        "with_a1": _aggregate_scores(a1_scores),
        "n_scored": len(base_scores),
    }


def score_combined_direction(
    src_m,
    dst_m,
    src_idx,
    dst_idx,
    g_src,
    g_dst,
    curve,
    fit_targets,
    score_targets,
    chrom,
    tss,
    chrom_idx,
    ctrl_means,
    symbols,
    symbol_cols,
) -> dict[str, Any]:
    """Score v2/+A1/+A2/+A1+A2 all on the same target subset, for a fair combined comparison."""
    bin_effects = model.fit_bin_effects_one_line(src_m, src_idx, chrom, tss, chrom_idx, fit_targets)
    denom = float(np.dot(g_src, g_src))
    betas_by_bin: dict[int, list[float]] = {i: [] for i in range(len(model.STRENGTH_BINS) - 1)}
    for t in fit_targets:
        if t not in src_idx:
            continue
        y = src_m[src_idx[t]]
        beta_x = float(np.dot(y, g_src) / denom) if denom > 0 else 1.0
        s_val = model.strength(y)
        for i in range(len(model.STRENGTH_BINS) - 1):
            if model.STRENGTH_BINS[i] <= s_val < model.STRENGTH_BINS[i + 1]:
                betas_by_bin[i].append(beta_x)
                break
    beta_curve = [float(np.mean(v)) if v else 1.0 for v in betas_by_bin.values()]

    def beta_for(s_val):
        """Look up the A2 beta for a strength value."""
        for i in range(len(model.STRENGTH_BINS) - 1):
            if model.STRENGTH_BINS[i] <= s_val < model.STRENGTH_BINS[i + 1]:
                return beta_curve[i]
        return beta_curve[-1]

    rows: dict[str, list[dict[str, float]]] = {"v2": [], "a1": [], "a2": [], "a1_a2": []}
    for t in score_targets:
        if t not in src_idx or t not in dst_idx:
            continue
        src_v, true = src_m[src_idx[t]], dst_m[dst_idx[t]]
        s_val = model.strength(src_v)
        lam = model.lambda_for(s_val, curve)
        s = src_v - g_src
        kd_j, kd = None, 0.0
        if t in symbols:
            kd_j = symbols.index(t)
            kd = model.knockdown_delta(t, symbol_cols, ctrl_means)

        p_v2 = g_dst + lam * s
        if kd_j is not None:
            p_v2 = p_v2.copy()
            p_v2[kd_j] = g_dst[kd_j] + lam * s[kd_j] + kd
        rows["v2"].append(top200_scores(p_v2, true))

        neighbor_terms = model.neighbor_delta_terms(t, chrom, tss, chrom_idx, bin_effects, kd_j)
        p_a1 = p_v2.copy()
        for gi, eff in neighbor_terms:
            p_a1[gi] += eff
        rows["a1"].append(top200_scores(p_a1, true))

        beta = beta_for(s_val)
        p_a2 = beta * g_dst + lam * s
        if kd_j is not None:
            p_a2 = p_a2.copy()
            p_a2[kd_j] = beta * g_dst[kd_j] + lam * s[kd_j] + kd
        rows["a2"].append(top200_scores(p_a2, true))

        p_a1a2 = p_a2.copy()
        for gi, eff in neighbor_terms:
            p_a1a2[gi] += eff
        rows["a1_a2"].append(top200_scores(p_a1a2, true))

    return {k: _aggregate_scores(v) for k, v in rows.items()} | {"n_scored": len(rows["v2"])}


def score_a2_direction(
    src_m,
    dst_m,
    src_idx,
    dst_idx,
    g_src,
    g_dst,
    curve,
    fit_targets,
    score_targets,
    ctrl_means,
    symbols,
    symbol_cols,
) -> dict[str, Any]:
    """Fit A2 on fit_targets, score baseline vs +A2 on score_targets, one transfer direction."""
    denom = float(np.dot(g_src, g_src))
    betas_by_bin = {i: [] for i in range(len(model.STRENGTH_BINS) - 1)}
    for t in fit_targets:
        if t not in src_idx:
            continue
        y = src_m[src_idx[t]]
        s_val = model.strength(y)
        beta_x = float(np.dot(y, g_src) / denom) if denom > 0 else 1.0
        for i in range(len(model.STRENGTH_BINS) - 1):
            if model.STRENGTH_BINS[i] <= s_val < model.STRENGTH_BINS[i + 1]:
                betas_by_bin[i].append(beta_x)
                break
    beta_curve = [float(np.mean(v)) if v else 1.0 for v in betas_by_bin.values()]

    def beta_for(s_val):
        """Look up the A2 beta for a strength value."""
        for i in range(len(model.STRENGTH_BINS) - 1):
            if model.STRENGTH_BINS[i] <= s_val < model.STRENGTH_BINS[i + 1]:
                return beta_curve[i]
        return beta_curve[-1]

    base_scores, a2_scores = [], []
    for t in score_targets:
        if t not in src_idx or t not in dst_idx:
            continue
        src_v, true = src_m[src_idx[t]], dst_m[dst_idx[t]]
        s_val = model.strength(src_v)
        lam = model.lambda_for(s_val, curve)
        s = src_v - g_src
        kd_j, kd = None, 0.0
        if t in symbols:
            kd_j = symbols.index(t)
            kd = model.knockdown_delta(t, symbol_cols, ctrl_means)
        pred_base = g_dst + lam * s
        if kd_j is not None:
            pred_base = pred_base.copy()
            pred_base[kd_j] = g_dst[kd_j] + lam * s[kd_j] + kd
        base_scores.append(top200_scores(pred_base, true))
        beta = beta_for(s_val)
        pred_a2 = beta * g_dst + lam * s
        if kd_j is not None:
            pred_a2 = pred_a2.copy()
            pred_a2[kd_j] = beta * g_dst[kd_j] + lam * s[kd_j] + kd
        a2_scores.append(top200_scores(pred_a2, true))
    return {
        "beta_curve": beta_curve,
        "baseline": _aggregate_scores(base_scores),
        "with_a2": _aggregate_scores(a2_scores),
        "n_scored": len(base_scores),
    }
