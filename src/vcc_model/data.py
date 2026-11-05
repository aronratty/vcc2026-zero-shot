"""Pure I/O: read challenge/reference files into arrays and dicts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load config.yaml into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def read_gene_column(path: str | Path) -> list[str]:
    """Read a one-column gene/target CSV, dropping its header if it has one."""
    with open(path) as f:
        rows = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    if rows and rows[0].lower() in {"gene_name", "target_gene", "gene"}:
        return rows[1:]
    return rows


def load_xatlas_cache(cache_dir: str | Path) -> dict[str, Any]:
    """Load the compact X-Atlas/Orion delta/control/gene-axis cache from disk."""
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
    """Load the challenge-axis-restricted K562 GWPS CD-coefficient NPZ."""
    k = np.load(path)
    return {
        "target_gene": k["target_gene"].astype(str),
        "effect": k["effect"].astype(np.float32).T,
        "challenge_gene_idx": k["challenge_gene_idx"].astype(int),
    }


def load_tss_table(path: str | Path) -> dict[str, tuple[str, int]]:
    """Load the gene_name -> (chrom, tss) table GENCODE extraction produced."""
    out: dict[str, tuple[str, int]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["gene_name"]] = (row["chrom"], int(row["tss"]))
    return out
