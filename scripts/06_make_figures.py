"""Render docs/figures PNGs from scripts/03_fit_lolo.py's report JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_lambda_curve(curve: list[dict], out_path: Path) -> None:
    """Plot lambda (transfer weight) against perturbation strength bin."""
    labels = [f"{int(r['lo'])}-{'inf' if r['hi'] == float('inf') else int(r['hi'])}" for r in curve]
    values = [r["lambda_mean"] for r in curve]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(len(values)), values, marker="o", color="#2b6cb0")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("perturbation strength bin (#genes with |delta| > 0.1)")
    ax.set_ylabel("lambda (specific-effect transfer weight)")
    ax.set_title("HCT116<->HEK293T transfer weight by strength")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_neighbor_bins(bin_gate: list[dict], out_path: Path) -> None:
    """Plot the neighbor-vs-control delta difference with its 95% CI, per bin and line."""
    bins = sorted({row["bin"] for row in bin_gate}, key=lambda b: int(b.split("-")[0]))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, line in enumerate(["HCT116", "HEK293T"]):
        rows = [r for r in bin_gate if r["line"] == line]
        rows = sorted(rows, key=lambda r: int(r["bin"].split("-")[0]))
        x = np.arange(len(rows)) + (i - 0.5) * 0.15
        means = [r["diff_mean"] for r in rows]
        lo = [r["diff_mean"] - r["diff_ci_lo"] for r in rows]
        hi = [r["diff_ci_hi"] - r["diff_mean"] for r in rows]
        ax.errorbar(x, means, yerr=[lo, hi], fmt="o", label=line, capsize=3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=0, fontsize=8)
    ax.set_xlabel("TSS distance to target (bp)")
    ax.set_ylabel("neighbor - matched control delta (95% CI)")
    ax.set_title("A1: genomic-neighbor silencing by distance bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation(combined_mean: dict, out_path: Path) -> None:
    """Bar-chart v2/+A1/+A2/+A1+A2 MAE and sign agreement on the same target subset."""
    variants = ["v2", "+A1 (ships)", "+A2 (dropped)", "+A1+A2"]
    keys = ["v2", "a1", "a2", "a1_a2"]
    colors = ["#718096", "#2f855a", "#c53030", "#718096"]
    mae = [combined_mean[k]["mae_top200"] for k in keys]
    sign = [combined_mean[k]["sign_top200"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))
    axes[0].bar(variants, mae, color=colors)
    axes[0].set_title("top-200 MAE (lower is better)")
    axes[0].set_ylim(min(mae) * 0.98, max(mae) * 1.02)
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(variants, sign, color=colors)
    axes[1].set_title("top-200 sign agreement (higher is better)")
    axes[1].set_ylim(min(sign) * 0.98, max(sign) * 1.02)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_terms(out_path: Path) -> None:
    """Waterfall-plot one illustrative gene's cumulative g + lambda*s + A1 delta."""
    labels = ["start", "g\n(generic)", "+ lambda*s\n(specific transfer)", "+ A1\n(neighbor silencing)"]
    steps = [0.0, -0.003, -0.02, -0.033]
    cumulative = np.cumsum(steps)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bottoms = [0] + list(cumulative[:-1])
    colors = ["#a0aec0", "#2b6cb0", "#2f855a", "#c53030"]
    ax.bar(labels, steps, bottom=bottoms, color=colors)
    for i, (b, s) in enumerate(zip(bottoms, steps)):
        if i > 0:
            ax.text(i, b + s / 2, f"{s:+.3f}", ha="center", va="center", fontsize=8, color="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(cumulative.min() * 1.25, 0.01)
    ax.set_ylabel("PIE log1p delta (illustrative values)")
    ax.set_title("Additive model terms for one example gene")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate all docs/figures PNGs from a LOLO report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="out/lolo_report.json")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_lambda_curve(report["lambda_curve"], out_dir / "lambda_curve.png")
    plot_neighbor_bins(report["neighbor_bin_gate"], out_dir / "a1_neighbor_bins.png")
    plot_ablation(report["combined_ablation"]["mean_across_4_cells"], out_dir / "ablation_chart.png")
    plot_model_terms(out_dir / "model_terms_example.png")
    print(f"wrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
