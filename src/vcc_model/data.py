"""Load challenge controls and the compact reference-dataset caches.

Pure I/O: everything here reads files into arrays/dicts. Modeling logic
(generic response, lambda transfer, knockdown prior, neighbor term) lives
in reference.py and model.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def read_gene_column(path: str | Path) -> list[str]:
    """Read a headerless-or-headered single-column CSV (gene/target list)."""
    with open(path) as f:
        rows = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    if rows and rows[0].lower() in {"gene_name", "target_gene", "gene"}:
        return rows[1:]
    return rows


def load_xatlas_cache(cache_dir: str | Path) -> dict[str, Any]:
    """Compact PIE-processed X-Atlas/Orion cache: per-perturbation pseudobulk
    log1p deltas on a fixed gene axis, plus per-context control means.

    Expects: delta_p.npy, ctrl_means.npy, ctx_ids.npy, pert_ids.npy,
    meta.json, gene_metadata.parquet (see README "Reproduce" for how this
    cache is obtained — it is not committed to this repo).
    """
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    gmeta = pd.read_parquet(cache_dir / "gene_metadata.parquet")
    return {
        "meta": meta,
        "ctx_ids": np.load(cache_dir / "ctx_ids.npy"),
        "pert_ids": np.load(cache_dir / "pert_ids.npy"),
        "delta_p": np.load(cache_dir / "delta_p.npy", mmap_mode="r"),
        "ctrl_means": np.load(cache_dir / "ctrl_means.npy", mmap_mode="r"),
        "gene_metadata": gmeta,
    }


def load_k562_gwps(path: str | Path) -> dict[str, Any]:
    """Compact Replogle 2022 K562 genome-wide CRISPRi CD-coefficient matrix,
    restricted to the challenge gene axis (see scripts/02_build_reference_effects.py)."""
    k = np.load(path)
    return {
        "target_gene": k["target_gene"].astype(str),
        "effect": k["effect"].astype(np.float32).T,
        "challenge_gene_idx": k["challenge_gene_idx"].astype(int),
    }


def load_tss_table(path: str | Path) -> dict[str, tuple[str, int]]:
    """gene_name -> (chrom, tss) from scripts/01_download_references.py's GENCODE extract."""
    out: dict[str, tuple[str, int]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["gene_name"]] = (row["chrom"], int(row["tss"]))
    return out
