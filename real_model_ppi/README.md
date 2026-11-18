# Released-predictor test

This folder holds the test with three released interaction predictors that is
reported in Results ("Exclusion can reverse an advantage but need not do so"),
in Methods ("Released interaction predictors") and in Extended Data Table 5.

TUnA-R, D-SCRIPT and Topsy-Turvy, as released, score the same positive pairs and
candidate negatives drawn from the eligible region O and the excluded region U of
STRING physical v12.0 (combined score ≥ 700) under the configuration-model
prefilter at the source ratio (τ = 2). The declared target has 34,347,139
non-edge pairs, 92.04% of them excluded (*W* = 0.9204). The primary comparison is
D-SCRIPT versus TUnA-R; Topsy-Turvy was added after the primary result was known
(the second deviation in Methods).

## Layout

| Path | Contents |
|---|---|
| `DESIGN.md` | Design, gates and reading rule, fixed before any study pair was scored, with four dated amendments. Amendment 3 (one-sided D-SCRIPT gate) and Amendment 4 (Topsy-Turvy) are the two deviations reported in Methods. |
| `scripts/fetch_inputs.py` | Downloads the label-free inputs and writes a manifest of URLs, sizes and sha256 hashes. |
| `scripts/build_target.py` | Builds the target, the regions, the samples of 10,000 positives and 10,000 pairs from each of O and U, and the gate inputs. No model is run. |
| `scripts/count_o_u.py` | Label-free counts used to plan the test (`results/label_free_counts.txt`). |
| `scripts/hpc/` | Cluster setup, gates and scoring. `setup.sh`, `resume1.sh`, `resume2.sh` and `run_all.sbatch`: gate G1 for TUnA-R and D-SCRIPT (job 4598207, stopped at the gate). `run_main.sbatch` with `gate_check_a3.py`: amended gate, then the study pairs (job 4610895). `setup_tt.sh` and `run_tt.sbatch`: Topsy-Turvy (job 4613624). `make_conn*.py` build the scripts that uploaded these files and submitted the jobs; `unpack_fetch.py` unpacks the returned outputs and checks their hashes. |
| `scripts/analyze.py` | Primary comparison: `results/analysis_4610895.json`. |
| `scripts/analyze_tt.py` | All three comparisons, Extended Data Table 5: `results/analysis_tt_4613624.json`. |
| `smoke_inputs/` | Synthetic sequences for the smoke tests (not biological data). |
| `results/target/` | `target_summary.json` (counts and *W*), the sampled pairs `pairs_{positives,O,U}.tsv`, their sequences and the gate subsets. |
| `results/scores_4610895/`, `results/scores_with_tt/` | Per-pair scores from jobs 4610895 and 4613624. The TUnA-R and D-SCRIPT files are identical in both. |
| `results/downloads_manifest.json`, `results/tt_inputs/` | Sources and hashes of the downloads. |
| `cluster_outputs/` | Logs and outputs of the three jobs as returned from the cluster: the gate records (`out/gate/` is the failed G1 gate; `out/gate_a3/` and `out/gate_tt/` passed), G1 and command-line checks, study scores, environment freezes, load tests, and the hash lists `out/OUTPUT_SHA256` and `out/OUTPUT_SHA256_TT`. |

## Reproduce Extended Data Table 5 from the saved scores

From this folder, with Python 3 and NumPy:

```sh
python scripts/analyze.py results/target results/scores_4610895 analysis_4610895.json
python scripts/analyze_tt.py results/target results/scores_with_tt results/analysis_4610895.json analysis_tt_4613624.json
```

Both outputs are identical to the files in `results/`, and `analyze_tt.py` stops
unless its recomputed primary comparison matches `analysis_4610895.json` exactly.
Each takes seconds. To check the returned cluster outputs against their recorded
hashes:

```sh
cd cluster_outputs/out && sha256sum -c OUTPUT_SHA256 OUTPUT_SHA256_TT
```

## Rerun the scoring

`fetch_inputs.py` and `build_target.py` run on a workstation; `build_target.py`
also needs `pyarrow`, the STRING files `data/string_physical_9606.txt.gz` and
`data/string_phys_700.tsv` at the repository root (from `hpc/fetch_networks.sh`),
and the downloads under `real_model_ppi/downloads/`. Scoring needs a GPU node and
about 10 GB of disk for the environments and downloads. The cluster scripts were
written for one Slurm cluster: before reuse, set the working directory `R`, the
partition, the excluded nodes, the `#SBATCH` output path and the mail address.
Setup downloads pinned releases and checks their hashes, and no study pair is
scored unless the gate passes.

## Placeholders in this public copy

Only paths, host names and account names were replaced; nothing else was changed.

- `$HOME` in scripts and `<HPC_HOME>` in recorded outputs: the cluster home directory.
- `<user>@<hpc-login-host>`, `<hpc-host>`, `<storage-host>` and `<HOME_ROOT>`: the login host, the hosts that ran the setup, the storage server and the root of the home directories.
- `"$USER"` in `squeue -u`, and `<email>` in `#SBATCH --mail-user`.
- `BENV=$HOME/env/bin/python` in `setup.sh`: an existing Python used only to create a virtual environment; if it is absent, the script uses `python3`.
- `<REPO_ROOT>` in `count_o_u.py`: the repository root.

No file listed in `cluster_outputs/out/OUTPUT_SHA256` or `OUTPUT_SHA256_TT` was changed.

## Sources and licences

The code in this folder is under the repository's MIT licence. Included data
taken from third-party releases keep their licences:

- STRING v12.0 (CC BY 4.0): network and protein sequences.
- SNOOPPI (Zenodo record 22116351, CC BY 4.0): released TUnA-R scores and sequences of the 300 G1 reference pairs.
- D-SCRIPT and Topsy-Turvy (github.com/samsledje/D-SCRIPT, MIT): released test pairs and sequences used in the G1 and command-line checks.
- TUnA (github.com/Wang-lab-UCSD/TUnA, MIT) and TUnA-R (github.com/young-su-ko/TUnA-R, MIT): the Bernett gold standard as distributed with them, used to remove released pairs from the target.

Model weights are not included; `scripts/hpc/setup.sh` and `setup_tt.sh` download
them from their sources and record their hashes.
