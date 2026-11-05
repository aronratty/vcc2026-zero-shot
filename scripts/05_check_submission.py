"""Run submission integrity checks on a decoded prediction file.

python scripts/05_check_submission.py out/predictions.h5ad --contexts D,E,F
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vcc_model import checks, data


def main() -> None:
    """Run checks.py's integrity checks on a prediction file and print the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("h5ad_path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--contexts", required=True, help="comma-separated, e.g. D,E,F")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths, dec_cfg = cfg["paths"], cfg["decode"]
    contexts = args.contexts.split(",")

    challenge_genes = data.read_gene_column(paths["challenge_genes"])
    targets = data.read_gene_column(paths["pert_counts"])
    control_paths = {c: Path(paths["controls_dir"]) / f"context_{c}.h5ad" for c in contexts}

    report = {
        "label_integrity": checks.label_integrity(
            args.h5ad_path, set(contexts), dec_cfg["cells_per_perturbation"]
        ),
        "nonneg_integer_counts": checks.nonneg_integer_counts_sampled(args.h5ad_path),
    }
    ratio_report = checks.target_gene_ratio_report(
        args.h5ad_path, control_paths, targets, challenge_genes, dec_cfg["cells_per_perturbation"]
    )
    report["target_gene_ratio_summary"] = ratio_report["summary"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
