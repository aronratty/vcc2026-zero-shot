# Methods

## Units and decoding

The reference cache (PIE-processed X-Atlas/Orion) stores deltas in
log1p(CP10k)-normalized space, not raw counts. Converting a log1p delta to
a count-space fold change with a naive `exp(delta)` understates every
change for low/medium-expression genes — `expm1(mean(log1p(x)))` is below
the mean count-scale expression in general (Jensen's inequality). For each
challenge context, the decoder computes `m_ctx` as the mean log1p(CP10k)
profile over that context's own non-targeting cells, then:

```text
fc = expm1(m_ctx + delta) / expm1(m_ctx)
```

Non-target genes with `expm1(m_ctx) < 0.01` CP10k use `fc = 1` (too sparse
to trust); other non-target fold changes are clipped to `[0.05, 20]`.
Target genes use the fixed challenge prior `fc = 0.10` exactly (Arc's
disclosed >=80% guide-selection rule; not fit locally — sensitivity values
0.05 and 0.20 are recorded in config.yaml but not used). Counts are decoded
sparsely: Binomial thinning for `fc < 1`, `count + Poisson(count*(fc-1))`
for `fc > 1`.

## Structured effect model

The prediction is `g + lambda(strength)*s + KD + A1`.

- `g`: the mean perturbation delta per reference line, averaged across
  HCT116 and HEK293T (X-Atlas/Orion PIE cache).
- `s`: the reference delta minus `g`, averaged where both reference lines
  exist for a target.
- `lambda(strength)`: fit by perturbation-strength bin (`#genes with
  |delta| > 0.1`) on HCT116<->HEK293T perturbation-held-out transfer.
- `KD`: the explicit target-gene prior; the decoder enforces its exact
  count-space fold change regardless of what this term's PIE-space value
  implies (the term exists for auditability of the delta file).
- K562-only perturbations (no HCT116/HEK293T reference) use signed CD
  coefficients, rank-remapped onto the HCT116 delta-magnitude scale
  (`reference.K562QuantileMap`) — K562's processed values are
  characteristic-direction coefficients, not log1p deltas, so only their
  *rank* transfers.
- `A1`: additive genomic-neighbor silencing (see "A1/A2 ablations" below).
  Network propagation (an HCT116 co-response graph) and a genomic-neighbor
  term were both explored in earlier iterations; propagation was dropped
  (held-out evaluation selected zero propagation weight) before A1 was
  re-derived with proper TSS coordinates and a leakage-controlled gate.

## A1/A2 ablations (final-day, 2026-11-05)

Both experiments used the `g + lambda*s + KD` baseline and the 3,277
perturbations shared between HCT116 and HEK293T. New parameters were fit
on an even-named half of shared targets and scored on the odd-named half
(both transfer directions, both fit/score orders — 4 scoring cells) to
control for target-identity leakage. Scores are top-200 true-DE-gene MAE
and sign agreement. See RESULTS.md for the full numbers.

**A1 (CRISPRi genomic-neighbor silencing):** CRISPRi represses chromatin
around a target's promoter, so genes whose promoters sit close to the
target's can be knocked down too. TSS coordinates came from GENCODE v44
basic. Same-chromosome neighbor genes were binned by TSS distance and
compared to distance- and control-expression-matched control genes
(same chromosome, 1-5 Mb away). The 0-1kb, 1-10kb, and 10-50kb bins show a
consistent drop in both lines (bootstrap 95% CI excludes 0); the 50-200kb
bin does not (HEK293T's CI crosses 0) and is not used. A bin effect was fit
on the full shared_with_tss set for the production model (the even/odd
split is validation-only). **Ships** — improves both MAE and sign
agreement in all 4 scoring cells, though the effect is small (it touches
522 of 5,559,900 target/gene prediction slots in the final submission).

**A2 (strength-scaled generic response):** the idea that a stronger
perturbation (more responding genes) should scale up the generic response
`g`, and that perturbation *strength* might transfer across cell lines
better than *which genes move*. Strength correlation between HCT116 and
HEK293T was 0.81 (Spearman, well above a 0.3 go/no-go gate), so a
per-strength-bin `beta` was fit (least-squares coefficient of each
perturbation's own delta on its own line's `g`) and used to replace flat
`g` with `beta(strength) * g`. **Dropped** — it improves both metrics only
in the HEK->HCT direction and worsens both in HCT->HEK.

## Local-test limitations

There is no perturbed D/E/F ground truth available for this task, so the
real challenge-context biology and the official (and, for 2026, unpublished)
leaderboard metric cannot be validated locally. HCT116/HEK293T LOLO is a
same-assay transfer test between two reference lines, not a test of the
actual zero-shot D/E/F contexts — it is the most rigorous proxy available,
not a guarantee. See README "Limitations and next steps".
