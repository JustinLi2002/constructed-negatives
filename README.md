# Constructed negatives determine the population a benchmark can rank over

Analysis code for Paper 2, using reconstructed data and predictions from
[`ptm-audit`](https://github.com/JustinLi2002/ptm-audit). License: **MIT**.
The development repository remains private during revision. This checkout
does not constitute a public release or a new Paper 2 Zenodo snapshot.

## What is measured

`W` is the excluded candidate-negative mass in a specified target. The
half-mass condition only permits a sufficiently large observed gap to satisfy
a worst-case bound; it does not certify an actual ranking. The positive
distribution and compared scoring functions stay fixed within a decomposition.
Reference-proteome coverage and reconstruction effects are separate outputs.
The latter compare actual ESM2 augmentation against baseline within each
split, with the reconstruction's own weights. They establish neither
DeepMVP–MusiteDeep rankings nor full-proteome performance.

## Installation

Python 3.10+ and `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy pandas scipy scikit-learn matplotlib
```

On Windows activate with `.venv\Scripts\Activate.ps1`. Existing cluster
environments can be used directly. No GPU or retraining is needed for the
archived-prediction analyses. Reanalysis used Python 3.12.2, NumPy 1.26.4,
pandas 2.2.2, SciPy 1.13.1, scikit-learn 1.5.1 and Matplotlib 3.9.2.

## Quick start and expected output

```sh
python prediction_files.py --self-test
python scope_reanalysis.py --self-test
python support_unit_audit.py --self-test
python verify_scope_review.py
python support_unit_audit.py
python make_figures.py
```

Expected: condition-isolation checks, four scope tests and the support-unit
counterexample pass; verification checks 792 run/depth rows, 48 input hashes
and all 16 rows of Table 1. Five PNGs are written to `figures/`: three main
figures and two Extended Data figures. These saved-result checks/redraws take
seconds on a workstation, without external prediction files.

## External data

PTM data: [Zenodo 21670043](https://doi.org/10.5281/zenodo.21670043).
Earlier audit v1.2: [Zenodo 22102323](https://doi.org/10.5281/zenodo.22102323).
An archive DOI does not establish that every trained score file is included.
Before a rerun, check the 48 filenames and hashes in
`results_scope_review_manifest.json` against retrieved predictions. Missing
files must be obtained from the matching audit release; do not substitute
shuffled controls or silently regenerate unmatched models.

Required base layout:

```text
HRP/
  rebuilt/{task}_all.tsv
  pdisjoint_runs_v2/{task}__replica__baseline__split{0,1,2}__on_rebuilt.pred.tsv
  pdisjoint_runs_v2/{task}__replica__ppi__split{0,1,2}__esm__on_rebuilt.pred.tsv
  deepmvp/DeepMVP/data/swiss_prot_human_20190214.fasta
```

The within-O shift diagnostic also requires corresponding `__on_replica`
files. Scores must contain `protein`, `pos`, `y`, `y_pred`, `init_0`, `init_1`.
Missing or unmatched sites cause an error. A different snapshot is a new
analysis, not a reproduction of the archived numbers.

PPI inputs are STRING physical v12.0, BioGRID 4.4.246 and HIPPIE's 2026-08-28
snapshot. Fetch commands are in `hpc/fetch_networks.sh`; raw downloads are not
bundled. Saved `clusterC/results_uniform/*_per_protein.tsv` files suffice for
the node-versus-pair audit and plotted summaries. UPNA-PPI's negative set is
in [its source repository](https://github.com/alxndgb/UPNA-PPI); its proprietary
positive graph is not supplied here.

## Script-to-result map

| Script | Output / manuscript role | Inputs and compute |
|---|---|---|
| `scope_reanalysis.py` | Table 1, Fig. 1b/2b/2c; per-run, per-depth, coverage, summary, hash manifest | FASTA, rebuilt sites, 48 prediction files; CPU/I/O, no training |
| `prediction_files.py` | Condition-aware indexing; duplicate rejection and shuffled-control isolation | Filenames; seconds |
| `measured_c.py` | Compact per-split regional effects | Rebuilt sites and predictions; CPU/I/O |
| `deployment_scope.py` | Reference screen (A) and conditional scenario (B), no measured C | FASTA and annotations; CPU/I/O |
| `ptm_guarantee.py`, `ranking_identification.py` | Published-gap scenarios and 42 channel comparisons, Fig. 2a | Earlier audit checkout; not reconstruction certificates |
| `metric_linearity.py` | AUROC/average-precision arithmetic diagnostic, ED Fig. 2b | Rebuilt sites and baseline predictions; CPU/I/O |
| `shift_check_ptm.py` | Per-split within-O shift | Replica/rebuilt evaluations; CPU/I/O |
| `shift_check.py` | Controlled PPI decomposition, Fig. 1c | STRING graph and controlled features; model fitting |
| `identification_bound.py` | Synthetic threshold/shape experiments, ED Fig. 1 | Self-contained; full twenty-seed grid includes fitting |
| `cl3_exclusion.py` | Eligible partners by degree, Fig. 3 / ED Table 1 | Edge list; graph size determines time/memory |
| `support_unit_audit.py` | `results_support_unit.tsv`, pair coverage and inclusive cutoff | Saved per-protein counts; seconds |
| `ddb_sampling.py` | Degree-composition taxonomy evidence | Edge list; graph-dependent |
| `ppi_consequence.py` | Controlled PPI effects, ED Fig. 2a; optional per-seed output | Edge list; full grid includes model fitting |
| `threshold_tradeoff.py` | Contamination sensitivity, ED Fig. 2c | Rebuilt sites and external `fn_sensitivity.py` |
| `reversal_bound.py` | Helpers imported by PPI decomposition | Imported, not a separate experiment |
| `make_figures.py` | All five PNGs | Saved tables; seconds |
| `verify_scope_review.py` | Numeric/provenance/Table 1 checks | Saved outputs; seconds |

Full external-data runtimes have not been benchmarked consistently. A
synthetic-sweep runtime should not be interpreted as a graph-processing time.

## Recompute PTM analyses

```sh
python scope_reanalysis.py --base /path/to/HRP --out results_scope_review
python measured_c.py --root /path/to/HRP/pdisjoint_runs_v2 --data /path/to/HRP/rebuilt
PTM_AUDIT_BASE=/path/to/HRP python deployment_scope.py
python shift_check_ptm.py --root /path/to/HRP/pdisjoint_runs_v2 --data /path/to/HRP/rebuilt --out results_shift_review.tsv
python metric_linearity.py --root /path/to/HRP/pdisjoint_runs_v2 --data /path/to/HRP/rebuilt > results_metric_review.txt
```

Table 1 averages three separately evaluated split-specific ensemble statistics.
Seven tasks reverse sign in all three splits; phosphorylation Y retains its
sign. Among six initialization-specific comparisons, ubiquitination has one
exception. These are empirical observations, not confidence guarantees.
No prediction file supplies depth-zero negatives for a full-proteome claim.

`threshold_tradeoff.py` imports `fn_sensitivity.py` from the first audit,
archived at [Zenodo 21993819](https://doi.org/10.5281/zenodo.21993819). Retrieve
that component before running the threshold analysis.

## Current versus historical results

Active PTM outputs: `results_scope_review_*`, `results_shift_review.tsv` and
`results_metric_review.txt`. The 792-row table covers eight tasks, three splits,
three score versions and eleven minimum depths. Identities are checked within
each split; products of mean W and mean c are not used as mean effects.

**Superseded:** `results_measured_c.txt`, `results_deployment_scope.txt`,
`results_shift_check_ptm.txt`, `results_metric_linearity.txt` remain historical
records only. Old indexing conflated shuffled and actual augmentation; the
former could overwrite the latter. Old scope calculations also reused a fixed
c across changing targets. The corrected scripts prevent both errors. Old
measured-C percentages do not support the revised manuscript.

`sweep_T_20seed.tsv` stores individual synthetic replicates. The saved PPI
network sweep contains per-cell aggregates and reversal counts over twenty
seeds, not original individual score records. `ppi_consequence.py --sweep`
can regenerate per-seed records, but a fresh run must be labelled as such.
The synthetic implementation reconstructs an earlier unarchived feasibility
experiment; current saved outputs are the manuscript's numerical inputs.
