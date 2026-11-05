"""Restrict the raw Replogle K562 GWPS table to the challenge gene axis."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import yaml


def read_gene_column(path: Path) -> list[str]:
    """Read a one-column gene/target CSV, dropping its header if it has one."""
    with open(path) as f:
        rows = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    return rows[1:] if rows and rows[0].lower() in {"gene_name", "target_gene", "gene"} else rows


def build_k562_compact(
    raw_path: Path, challenge_genes_path: Path, pert_counts_paths: list[Path], out_path: Path
) -> dict:
    """Parse the raw K562 GWPS table and save a compact, challenge-axis-restricted NPZ."""
    challenge = read_gene_column(challenge_genes_path)
    gene_index = {gene: i for i, gene in enumerate(challenge)}
    wanted: set[str] = set()
    for p in pert_counts_paths:
        wanted |= set(read_gene_column(p))

    with gzip.open(raw_path, "rt") as handle:
        header = next(handle).rstrip("\n").split("\t")[1:]
        columns = []
        for j, label in enumerate(header, start=1):
            if "_" not in label or "NON-TARGETING" in label:
                continue
            target = label.split("_", 1)[1].rsplit("_P", 1)[0]
            if target in wanted:
                columns.append((j, target))
        row_indices, rows, total_rows = [], [], 0
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            total_rows += 1
            if fields[0] in gene_index:
                row_indices.append(gene_index[fields[0]])
                rows.append([float(fields[j]) for j, _ in columns])

    matrix = np.asarray(rows, dtype=np.float32)
    column_targets = np.asarray([t for _, t in columns], dtype="U32")
    unique_targets = list(dict.fromkeys(column_targets.tolist()))
    # a target can have multiple guides/columns; average them
    matrix = np.column_stack([matrix[:, column_targets == t].mean(axis=1) for t in unique_targets]).astype(
        np.float32
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        effect=matrix,
        challenge_gene_idx=np.asarray(row_indices, dtype=np.int32),
        target_gene=np.asarray(unique_targets, dtype="U32"),
    )
    return {
        "processed_perturbations": len(unique_targets),
        "mapped_measured_genes": len(row_indices),
        "unmapped_measured_genes": total_rows - len(row_indices),
    }


def main() -> None:
    """Build the compact K562 NPZ from config.yaml's paths, unless it already exists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]

    out_path = Path(paths["k562_gwps_compact"])
    if out_path.exists():
        print(f"already have {out_path}")
        return
    report = build_k562_compact(
        Path(paths["k562_gwps_raw"]), Path(paths["challenge_genes"]), [Path(paths["pert_counts"])], out_path
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
