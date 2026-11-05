"""Run the HCT116<->HEK293T LOLO benchmark and the A1/A2 ablation gates.

Reproduces the numbers in RESULTS.md "Final-day A1/A2 ablations": the
neighbor-silencing bin statistics, the even/odd-split fit/score for A1 and
A2 in both transfer directions, and the production lambda curve used by
scripts/04_predict.py. Writes out/lolo_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from vcc_model import benchmark, data, model, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="out/lolo_report.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]

    challenge_genes = data.read_gene_column(paths["challenge_genes"])
    xatlas = data.load_xatlas_cache(paths["xatlas_cache"])
    symbols, symbol_cols = reference.align_gene_axis(xatlas, challenge_genes)
    cols = np.asarray([symbol_cols[s][0] for s in symbols], dtype=int)
    h_names, H = reference.load_reference_line(xatlas, "hct116", cols)
    e_names, E = reference.load_reference_line(xatlas, "hek293t", cols)
    h_idx = {n: i for i, n in enumerate(h_names)}
    e_idx = {n: i for i, n in enumerate(e_names)}
    ctrl_means = xatlas["ctrl_means"]
    tss_table = data.load_tss_table(paths["tss_table"])
    ctrl_expr = reference.control_expression(xatlas, symbols, symbol_cols)
    chrom, tss, chrom_idx = reference.build_chrom_index(symbols, tss_table, ctrl_expr)

    g_h, g_e, g_common = model.generic_response(H, E)
    curve_he = model.fit_lambda_curve(H, E, g_h, g_e, h_names, e_names)
    curve_eh = model.fit_lambda_curve(E, H, g_e, g_h, e_names, h_names)
    curve_avg = model.average_curves(curve_he, curve_eh)

    shared = sorted(set(h_idx) & set(e_idx))
    shared_with_tss = sorted(t for t in shared if chrom.get(t) is not None)
    evens, odds = benchmark.even_odd_split(shared_with_tss)
    print(f"shared={len(shared)} shared_with_tss={len(shared_with_tss)} evens={len(evens)} odds={len(odds)}")

    bin_gate = benchmark.neighbor_bin_gate_stats(H, E, h_idx, e_idx, chrom, tss, chrom_idx, shared_with_tss)

    a1_results = {}
    for direction, (src_m, dst_m, src_i, dst_i, g_src, g_dst) in {
        "HCT->HEK": (H, E, h_idx, e_idx, g_h, g_e),
        "HEK->HCT": (E, H, e_idx, h_idx, g_e, g_h),
    }.items():
        for order, (fit_t, score_t) in {
            "evens_to_odds": (evens, odds),
            "odds_to_evens": (odds, evens),
        }.items():
            a1_results[f"{direction}_{order}"] = benchmark.score_a1_direction(
                src_m,
                dst_m,
                src_i,
                dst_i,
                g_src,
                g_dst,
                curve_avg,
                fit_t,
                score_t,
                chrom,
                tss,
                chrom_idx,
                ctrl_means,
                symbols,
                symbol_cols,
            )

    a1_ships = all(
        a1_results[k]["with_a1"]["mae_top200"] < a1_results[k]["baseline"]["mae_top200"]
        and a1_results[k]["with_a1"]["sign_top200"] > a1_results[k]["baseline"]["sign_top200"]
        for k in a1_results
    )

    rho, p, n_shared_strength = benchmark.strength_transfer_spearman(H, E, h_idx, e_idx)
    a2_results, a2_ships = {}, False
    if rho >= 0.3:
        all_shared_evens, all_shared_odds = benchmark.even_odd_split(shared)
        for direction, (src_m, dst_m, src_i, dst_i, g_src, g_dst) in {
            "HCT->HEK": (H, E, h_idx, e_idx, g_h, g_e),
            "HEK->HCT": (E, H, e_idx, h_idx, g_e, g_h),
        }.items():
            for order, (fit_t, score_t) in {
                "evens_to_odds": (all_shared_evens, all_shared_odds),
                "odds_to_evens": (all_shared_odds, all_shared_evens),
            }.items():
                a2_results[f"{direction}_{order}"] = benchmark.score_a2_direction(
                    src_m,
                    dst_m,
                    src_i,
                    dst_i,
                    g_src,
                    g_dst,
                    curve_avg,
                    fit_t,
                    score_t,
                    ctrl_means,
                    symbols,
                    symbol_cols,
                )
        a2_ships = all(
            a2_results[k]["with_a2"]["mae_top200"] < a2_results[k]["baseline"]["mae_top200"]
            and a2_results[k]["with_a2"]["sign_top200"] > a2_results[k]["baseline"]["sign_top200"]
            for k in a2_results
        )

    # Combined table: v2/+A1/+A2/+A1+A2 all scored on the SAME shared_with_tss
    # subset, so the bars are apples-to-apples (A1/A2 scored separately above
    # use different target populations — A1 needs a mapped TSS, A2 doesn't —
    # so their own baselines aren't comparable to each other directly).
    combined_results = {}
    for direction, (src_m, dst_m, src_i, dst_i, g_src, g_dst) in {
        "HCT->HEK": (H, E, h_idx, e_idx, g_h, g_e),
        "HEK->HCT": (E, H, e_idx, h_idx, g_e, g_h),
    }.items():
        for order, (fit_t, score_t) in {
            "evens_to_odds": (evens, odds),
            "odds_to_evens": (odds, evens),
        }.items():
            combined_results[f"{direction}_{order}"] = benchmark.score_combined_direction(
                src_m,
                dst_m,
                src_i,
                dst_i,
                g_src,
                g_dst,
                curve_avg,
                fit_t,
                score_t,
                chrom,
                tss,
                chrom_idx,
                ctrl_means,
                symbols,
                symbol_cols,
            )
    combined_mean = {
        variant: {
            m: float(np.mean([combined_results[k][variant][m] for k in combined_results]))
            for m in ["mae_top200", "sign_top200", "overlap_top200"]
        }
        for variant in ["v2", "a1", "a2", "a1_a2"]
    }

    report = {
        "lambda_curve": curve_avg,
        "neighbor_bin_gate": bin_gate,
        "a1": {"per_cell": a1_results, "ships": a1_ships},
        "a2": {
            "strength_spearman": rho,
            "spearman_p": p,
            "n_shared": n_shared_strength,
            "gate_passed": rho >= 0.3,
            "per_cell": a2_results,
            "ships": a2_ships,
        },
        "combined_ablation": {
            "n_shared_with_tss": len(shared_with_tss),
            "per_cell": combined_results,
            "mean_across_4_cells": combined_mean,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"A1 ships: {a1_ships}")
    print(
        f"A2 strength spearman: {rho:.4f} (gate {'passed' if rho >= 0.3 else 'failed'}); A2 ships: {a2_ships}"
    )
    print(f"combined ablation (mean across 4 cells): {json.dumps(combined_mean, indent=2)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
