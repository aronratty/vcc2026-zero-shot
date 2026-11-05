"""Build predicted deltas and decode them into a submission-ready AnnData of raw counts.

python scripts/04_predict.py --contexts D,E,F
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import yaml
from scipy import sparse

from vcc_model import data, decode, model, reference


def main() -> None:
    """Build the model's predicted deltas, decode them, and write a submission h5ad."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--contexts", required=True, help="comma-separated, e.g. D,E,F")
    parser.add_argument(
        "--targets", default=None, help="comma-separated subset of targets (default: all official)"
    )
    parser.add_argument("--cells-per-pert", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-output", default=None, help="also save the raw delta matrix here")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths, dec_cfg = cfg["paths"], cfg["decode"]
    contexts = args.contexts.split(",")
    cells_per_pert = args.cells_per_pert or dec_cfg["cells_per_perturbation"]
    output = Path(args.output or paths["predictions_out"])

    challenge_genes = data.read_gene_column(paths["challenge_genes"])
    final_targets = data.read_gene_column(paths["pert_counts"])
    if args.targets:
        final_targets = [t for t in args.targets.split(",") if t in final_targets]

    xatlas = data.load_xatlas_cache(paths["xatlas_cache"])
    symbols, symbol_cols = reference.align_gene_axis(xatlas, challenge_genes)
    cols = np.asarray([symbol_cols[s][0] for s in symbols], dtype=int)
    h_names, H = reference.load_reference_line(xatlas, "hct116", cols)
    e_names, E = reference.load_reference_line(xatlas, "hek293t", cols)
    h_idx = {n: i for i, n in enumerate(h_names)}
    e_idx = {n: i for i, n in enumerate(e_names)}
    ctrl_means = xatlas["ctrl_means"]

    k562 = data.load_k562_gwps(paths["k562_gwps_compact"])
    k562_map = reference.K562QuantileMap(k562, challenge_genes, symbols, H, h_idx)

    tss_table = data.load_tss_table(paths["tss_table"])
    ctrl_expr = reference.control_expression(xatlas, symbols, symbol_cols)
    chrom, tss, chrom_idx = reference.build_chrom_index(symbols, tss_table, ctrl_expr)

    g_h, g_e, g_common = model.generic_response(H, E)
    curve_he = model.fit_lambda_curve(H, E, g_h, g_e, h_names, e_names)
    curve_eh = model.fit_lambda_curve(E, H, g_e, g_h, e_names, h_names)
    curve_avg = model.average_curves(curve_he, curve_eh)

    shared_with_tss = sorted(t for t in (set(h_idx) & set(e_idx)) if chrom.get(t) is not None)
    bin_effects = model.fit_a1_bin_effects(H, E, h_idx, e_idx, chrom, tss, chrom_idx, shared_with_tss)
    print(
        f"A1 bin effects (production fit, n={len(shared_with_tss)}): "
        f"{ {f'{lo}-{hi}': v for (lo, hi), v in bin_effects.items()} }"
    )

    deltas, has_specific = model.build_all_predictions(
        final_targets,
        challenge_genes,
        symbols,
        g_common=g_common,
        H=H,
        E=E,
        h_idx=h_idx,
        e_idx=e_idx,
        curve=curve_avg,
        symbol_cols=symbol_cols,
        ctrl_means=ctrl_means,
        k562_map=k562_map,
        chrom=chrom,
        tss=tss,
        chrom_idx=chrom_idx,
        bin_effects=bin_effects,
        target_fc=dec_cfg["target_fc_prior"],
    )
    print(f"{sum(has_specific)}/{len(final_targets)} targets have a reference-line-specific effect")

    if args.model_output:
        Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.model_output,
            delta=deltas,
            target_gene=np.asarray(final_targets, dtype="U32"),
            gene=np.asarray(challenge_genes, dtype="U32"),
        )

    blocks, obs_rows = [], []
    for cidx, context in enumerate(contexts):
        control_path = Path(paths["controls_dir"]) / f"context_{context}.h5ad"
        for _target_idx, (target, decoded, rows) in enumerate(
            decode.decode_context(
                control_path,
                context,
                challenge_genes,
                final_targets,
                deltas,
                seed=dec_cfg["seed"] + cidx,
                cells_per_pert=cells_per_pert,
                target_fc=dec_cfg["target_fc_prior"],
                mu_floor=dec_cfg["mu_floor"],
            )
        ):
            blocks.append(decoded)
            obs_rows.extend(rows)

    X = sparse.vstack(blocks, format="csr", dtype=np.int32)
    import pandas as pd

    obs = pd.DataFrame(obs_rows, index=[f"{r['context']}_{i:08d}" for i, r in enumerate(obs_rows)])
    var = pd.DataFrame(index=pd.Index(challenge_genes, name="gene_name"))
    out = ad.AnnData(X=X, obs=obs, var=var)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output, compression=None)
    print(
        {
            "output": str(output),
            "cells": int(X.shape[0]),
            "genes": int(X.shape[1]),
            "nnz": int(X.nnz),
            "bytes": int(output.stat().st_size),
        }
    )


if __name__ == "__main__":
    main()
