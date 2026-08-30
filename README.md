# Constructed negatives determine the population a benchmark can rank over

Code for the paper of the same name, which spins out of
[`ptm-audit`](https://github.com/JustinLi2002/ptm-audit) (Li & Yu, *Negative
sampling determines the sign of protein-level feature contributions in
modification site prediction*).

**`HANDOVER_identification.md`, in this repository, is the working process
log, not part of the submission.** It is kept here for continuity between
work sessions and is not required to reproduce any result below.

## The claim

A benchmark that draws negatives only from entities carrying at least `T`
positives supplies no negatives at all from entities below `T`. The propensity
of appearing as a negative is exactly zero on that region, so this is a
**positivity violation, not covariate shift**: importance weighting has a zero
denominator, and no estimator using only that benchmark identifies the
deployment-distribution effect. Every diagnostic computable inside the
benchmark comes back clean, because the data carry no information about the
missing region for them to disagree with.

What remains is partial identification, in Manski's sense. This code
establishes the violation, measures its size on two real benchmarks, and
tests when a reconstruction can replace the worst case with a measurement.

## Requirements

Python 3.10+, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`. No
GPU and no cluster; the synthetic experiment and the figure script run in
minutes on a laptop.

## Two external data dependencies

Reviewers should know about these before trying to run everything below:

1. **The PTM-side scripts** (`ptm_guarantee.py`, `ranking_identification.py`,
   `measured_c.py`, `metric_linearity.py`, `deployment_scope.py`,
   `threshold_tradeoff.py`) read the reconstructed positive/negative sets and
   per-site predictions from the first paper's audit, not from anything in
   this repository. `ptm_guarantee.py` and `ranking_identification.py` take a
   `root` argument that is a checkout of `ptm-audit`; the rest read from a
   `--data`/`PTM_AUDIT_BASE` directory laid out the way that repository's
   `rebuilt/` is. **These will not run until `ptm-audit`'s reconstructions are
   themselves public — an open dependency, tracked in
   `HANDOVER_identification.md`.**
2. **The PPI-side scripts** (`cl3_exclusion.py`, `ddb_sampling.py`,
   `ppi_consequence.py`, `shift_check.py`) read an edge list such as
   `data/string_phys_700.tsv`. `data/` is gitignored here because the full
   STRING download is large, but the network itself is public: STRING
   physical links v12.0 (string-db.org), filtered to human, score >= the
   threshold named in the filename. BioGRID (release 4.4.246) and HIPPIE
   (v2.4) are used the same way for the cross-network robustness checks.
   UPNA-PPI's own released negatives are public at
   github.com/alxndgb/UPNA-PPI.

Everything else — the synthetic-population experiment, the figure script, and
the per-replicate result files these two families of scripts already
produced — is self-contained in this repository.

## Contents

| script | reproduces | needs |
|---|---|---|
| `identification_bound.py` | Synthetic populations: the exact stratum decomposition, and three fill-in rules for the strata a benchmark never supplies (endpoint, observed-trend, observed-curvature). Backs Fig. 1a/1c and the Methods residual claims. | nothing external |
| `ptm_guarantee.py` | The assumption-free guarantee (eq. 4/6) applied to the eight real PTM tasks: `W`, `Δ_O`, and whether the guarantee fires. Backs Fig. 1b, Table 1. | `ptm-audit` checkout |
| `ranking_identification.py` | The ranking-identification criterion (eq. 5) on the same eight tasks and the 42 pairwise feature-channel comparisons. Backs Fig. 2a. | `ptm-audit` checkout |
| `measured_c.py` | Measured *c* per task, holding the positive set fixed and stratifying only the negatives. Backs Fig. 2b, Table 1's *c* column. | `ptm-audit` rebuilt data |
| `deployment_scope.py` | The scope statistic *d_min*: the smallest annotation depth at which *W* falls to 0.5 or below. Backs Table 1's column (A). | `ptm-audit` rebuilt data |
| `metric_linearity.py` | The AUROC-vs-AUPRC residual comparison — why the exact decomposition holds under one metric and not the other. Backs Extended Data Fig. 2b. | `ptm-audit` rebuilt data |
| `shift_check.py` | Splits the benchmark's error into the unobserved-mass term and the within-*O* covariate-shift term, on the STRING >=700 network. Backs the Methods *Within-O shift* paragraph. Imports `cl3_exclusion`, `ppi_consequence`, and `reversal_bound`. | `data/string_phys_700.tsv` |
| `cl3_exclusion.py` | Which proteins can contribute a negative under UPNA-PPI's length-3-path prefilter, by degree. Backs Table 2, Extended Data Table 1. | an edge list (`edges.tsv`) |
| `ddb_sampling.py` | Whether DDBSampling's degree-balanced negative rule (the taxonomy's corrective case) still leaves a zero region. Backs the taxonomy table and the Discussion's "do not use a donor threshold as false-negative control" point. | an edge list |
| `ppi_consequence.py` | The controlled consequence experiment: real topology and real sampling rule, constructed features with known ground truth, so the effect is attributable to the rule and not an artefact. Backs Fig. 2c-style results and the "violation changes conclusions" section. | an edge list |
| `threshold_tradeoff.py` | The donor-threshold cost/benefit trade on one axis: `W_un(T)` against the expected false-negative rate `FN(T)`. Backs the "what the threshold was buying" figure. | `ptm-audit` rebuilt data |
| `make_figures.py` | All main and Extended Data figures, drawn only from `results_*.txt` already in this repository (the small numbers embedded at the top of the script are transcribed from those files and named there, so they can be checked rather than trusted). | nothing external once the `results_*.txt` files exist |
| `reversal_bound.py` | Not run standalone for its own named result; imported by `shift_check.py` for the split-effect identity. Kept for that reproducibility path. | nothing external |

`check_consistency.py` and `build_docx.js` are manuscript tooling, not
analysis code, and are documented in `HANDOVER_identification.md` rather than
here.

## Quick start

```
# self-contained
python identification_bound.py --sweep --seeds 20 --out sweep_T_20seed.tsv
python identification_bound.py --demo

# needs a ptm-audit checkout
python ptm_guarantee.py /path/to/ptm-audit
python ranking_identification.py /path/to/ptm-audit
python measured_c.py --root ~/HRP/pdisjoint_runs_v2 --data ~/HRP/rebuilt
python metric_linearity.py --root ~/HRP/pdisjoint_runs_v2 --data ~/HRP/rebuilt
PTM_AUDIT_BASE=/path/to/rebuilt python deployment_scope.py
PTM_AUDIT_BASE=/path/to/rebuilt python threshold_tradeoff.py

# needs a network edge list (data/ is gitignored; see "Two external data
# dependencies" above for where to get one)
python cl3_exclusion.py data/string_phys_700.tsv --out string700
python ddb_sampling.py data/string_phys_700.tsv --out string700_ddb
python ppi_consequence.py data/string_phys_700.tsv --seeds 5
python shift_check.py

# once the results_*.txt files above exist
python make_figures.py
```

Every script prints or writes a `results_*.txt`/`.tsv` file; those already in
this repository are the per-replicate records the manuscript's numbers are
computed from, kept so a reviewer can check a reported figure against the
run that produced it rather than re-running everything from scratch.

## What the results say

The decomposition `Δ_deploy = Σ_k w_k · Δ_k` is exact — residual at machine
precision in every configuration tested — which is what licenses everything
else. Three fill-in rules were tested for the strata a benchmark does not
supply, and the resolution that survives past a realistic threshold (not the
missing mass) is the binding constraint: see the manuscript's Results,
"Where shape restrictions can and cannot help", for the numbers.

## Provenance

The original `identification_bound.py` was written 2026-08-20 and was never
committed anywhere; it is gone. This is a re-implementation from the
feasibility table it produced (recorded in the handover) plus the first
paper's `analysis/synthetic_reversal.py`, and it is not seed-compatible with
the original — the structure reproduces (unobserved weights match to 0.01,
the decomposition is exact) but the validity column does not, and the
handover records that correction.
