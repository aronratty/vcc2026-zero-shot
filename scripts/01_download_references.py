"""Fetch GENCODE TSS coordinates automatically; print manual steps for the rest."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path
from urllib.request import urlretrieve

import yaml

GENCODE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.basic.annotation.gtf.gz"
)

MANUAL_STEPS = """
Two reference datasets are used by this pipeline and are not fetched by
this script — see README "Reproduce" for license terms:

1. X-Atlas/Orion (Xaira Therapeutics, 2025) HCT116/HEK293T Perturb-seq.
   License: CC BY-NC-SA (non-commercial). Source: figshare,
   https://doi.org/10.25452/figshare.plus.29190726
   This pipeline expects a compact *per-perturbation pseudobulk delta*
   cache derived from the raw release (the raw release is ~520 GB and is
   not downloaded directly) — see data.load_xatlas_cache() for the exact
   file schema expected at paths.xatlas_cache in config.yaml:
   delta_p.npy, ctrl_means.npy, ctx_ids.npy, pert_ids.npy, meta.json,
   gene_metadata.parquet.

2. Replogle et al. 2022 genome-wide CRISPRi Perturb-seq in K562
   (Cell, DOI 10.1016/j.cell.2022.05.013; bioRxiv 10.1101/2021.12.16.473013).
   Processed signatures are available via scPerturb
   (http://projects.sanderlab.org/scperturb/) or pertpy's
   `pertpy.data.replogle_2022_k562_gwps()` loader. This pipeline expects
   the raw per-gene-by-per-target text table (gzip TSV, target columns
   named "<guide>_<TARGET_SYMBOL>_P<n>") at paths.k562_gwps_raw in
   config.yaml; scripts/02_build_reference_effects.py then restricts it
   to the challenge gene axis.

Place both at the paths config.yaml points to, then run
`make build-reference`.
"""


def download_gencode(dest: Path) -> None:
    """Download the GENCODE v44 basic GTF if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already have {dest}")
        return
    print(f"downloading {GENCODE_URL} -> {dest}")
    urlretrieve(GENCODE_URL, dest)


def extract_tss(gtf_path: Path, challenge_genes_path: Path, out_path: Path) -> int:
    """Extract one TSS per challenge gene from the GTF, preferring canonical chromosomes."""
    with open(challenge_genes_path) as f:
        rows = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    challenge_genes = rows[1:] if rows and rows[0].lower() in {"gene_name", "gene"} else rows
    cset = set(challenge_genes)

    def is_canonical(chrom: str) -> bool:
        """True for chr1-22/X/Y, false for scaffolds/patches."""
        return re.fullmatch(r"chr([1-9]|1[0-9]|2[0-2]|X|Y)", chrom) is not None

    tss_by_symbol: dict[str, tuple[str, int, str]] = {}
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "gene":
                continue
            chrom, start, end, strand, attr = fields[0], int(fields[3]), int(fields[4]), fields[6], fields[8]
            m = re.search(r'gene_name "([^"]+)"', attr)
            if not m or m.group(1) not in cset:
                continue
            name = m.group(1)
            tss = start if strand == "+" else end
            rec = (chrom, tss, strand)
            prev = tss_by_symbol.get(name)
            if prev is None or (is_canonical(chrom) and not is_canonical(prev[0])):
                tss_by_symbol[name] = rec

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["gene_name", "chrom", "tss", "strand"])
        for name in challenge_genes:
            if name in tss_by_symbol:
                chrom, tss, strand = tss_by_symbol[name]
                w.writerow([name, chrom, tss, strand])
    return len(tss_by_symbol)


def main() -> None:
    """Download GENCODE, extract TSS coordinates, and print the manual data steps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    gtf_path = Path(cfg["paths"]["gencode_gtf"])
    download_gencode(gtf_path)
    mapped = extract_tss(gtf_path, Path(cfg["paths"]["challenge_genes"]), Path(cfg["paths"]["tss_table"]))
    print(f"TSS mapped for {mapped} challenge genes -> {cfg['paths']['tss_table']}")

    print(MANUAL_STEPS, file=sys.stderr)


if __name__ == "__main__":
    main()
