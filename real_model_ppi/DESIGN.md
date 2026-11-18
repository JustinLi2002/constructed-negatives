# Real-model check of the interaction consequence experiment: design

Drafted 2026-11-18. Status: approved by Justin on 2026-11-18, with computation on HRP where possible; Amendment 1 at the end was written before any new score was computed.
Everything in this file is fixed before any new score is read. If an item changes after that, the change is recorded below with its date and reason.

## 1. Question

Fix two released interaction predictors and one set of positive pairs. Change only which candidate negatives enter the evaluation. How does the AUROC gap between the two predictors change?

The manuscript currently has two separate pieces of evidence:
- released-data checks of protein coverage;
- a controlled-feature experiment showing that the ordering can reverse.

This study connects them with trained predictors.

The result is informative whether or not the ordering reverses, provided the intervals are narrow enough to state how much the benchmark gap is amplified or shrunk.

## 2. Models (pinned)

| Label | Model | Pin | Training data (as documented by the authors) |
|---|---|---|---|
| A | TUnA-R | Hugging Face `yk0/tuna-r-tuna`, revision `4e9ab56ffe2768daf632949543e5f440a4cd5a58` (2026-05-31; the June revision pinned by SNOOPPI's `tuna_june_checkpoint.py`; `pytorch_model.bin` 135,363,475 bytes; MIT). Embeddings: ESM-2 150M (`esm2_t30_150M_UR50D`), as in TUnA-R's `src/tuna/inference/embeddings.py`. The code commit is recorded at clone time. | Bernett human gold standard (model card); its negatives were chosen to account for node degree |
| B | D-SCRIPT | `human_v1.sav` (2,542,758 bytes; the original paper's human model) with the Bepler–Berger language model `lm_v1.sav` (238,258,695 bytes). The `dscript` package version is recorded at install time. | STRING v11 human positives with randomly paired negatives (original paper) |

**Not in the main comparison**
- PLM-interact is deferred because its cross-encoder needs a GPU.
- UPNA-PPI was trained on its own released negatives, so it is not compared.

**Attribution limit**
- A and B differ in architecture and training data as well as in how their negatives were built. A difference in how their gaps change therefore cannot be attributed to negative sampling alone.

**Direction hypothesis**
- This is stated in advance. It is not a gate and not a claim.
- Any advantage of B over A is larger on the eligible region O than on the excluded region U.
- Reason: randomly paired negatives allow a model to use protein-level degree, and the eligible region of this rule contains only pairs of degree-one and degree-two proteins.
- Whether either model actually relies on degree is tested by the scores, not assumed.

## 3. Target, rule and regions (primary analysis)

**Network**
- STRING physical v12.0 at combined score ≥ 700, as in the consequence experiment (`data/string_phys_700.tsv`).
- After symmetrising and removing self-loops and duplicate edges: 10,746 proteins and 86,519 edges.

**Rule**
- The configuration-model prefilter at the source ratio 0.064, with all ties at the cutoff retained, implemented by `cl3_exclusion.config_threshold`.
- This gives τ = 2: a non-edge pair is eligible when d_u·d_v ≤ 2.
- Known edges are never candidates.

**Label-free counts on the unrestricted candidate set**
- Computed 2026-11-18 by `scripts/count_o_u.py`, without any score; output in `results/label_free_counts.txt`.

| Quantity | Value |
|---|---|
| Non-edge pairs | 57,646,366 |
| Eligible (O) | 4,334,373 pairs, over 3,225 proteins (all proteins of degree 1 or 2) |
| Excluded (U) | 53,311,993 pairs |
| *W*, uniform weights | 0.9248 (the manuscript's measured range at this ratio is 0.888–0.937) |
| Positive edges with d_u·d_v ≤ 2 | 372 |
| Median degree product of positive edges | 1,869 |

**Declared target**
- The target is the candidate non-edge pairs that meet both conditions below, weighted uniformly. *W* is recounted on this target before any score is read.
  - (a) Both models can score them.
  - (b) The pair is not in either model's released datasets.
- **Scorable** means both proteins have a STRING v12.0 sequence of the 20 standard amino acids, of length 50–800.
  - 50–800 is the length range of D-SCRIPT's human training data. It also bounds CPU memory for D-SCRIPT's contact map.
  - *To confirm.*
- **Release exclusion** works on exact sequence identity. A pair is excluded if its unordered pair of sequences appears in any of these:
  - D-SCRIPT's `human_train.tsv` or `human_test.tsv`;
  - any Bernett partition released with TUnA (`Intra0`, `Intra1`, `Intra2`, positives and negatives).
  
  Test and validation pairs are excluded as well as training pairs. This loses few pairs and removes any doubt about early stopping.
- O and U are the target pairs inside and outside the rule's eligible region.

**Descriptive only**
- The share of target pairs whose two proteins both appear somewhere in either model's released data.
- The share of sampled O and U pairs with any STRING physical score in [400, 700). This is a proxy for undetected interactions, from the local `data/string_physical_9606.txt.gz`.

## 4. Positives

- Positives are STRING ≥ 700 edges, deduplicated as unordered pairs.
- They pass the same scorability filter and the same release exclusion.
- A uniform sample of 10,000 is used (seed 20261118), or all of them if fewer remain.
- Both models and both regions share the same positives.

## 5. Candidate negatives

- 10,000 pairs are drawn uniformly without replacement from target O, and 10,000 from target U.
- Seed 20261118. Rejection sampling against the edge set, the scorability filter and the release exclusion.

## 6. Estimands and the one primary result

**Definitions**
- AUROC_k(R) is the AUROC of model k for the shared positives against the sampled negatives of region R. Ties count one half.
- Δ_R = AUROC_B(R) − AUROC_A(R), that is, D-SCRIPT minus TUnA-R.
- Δ_target = (1 − *W*)Δ_O + *W*Δ_U, with *W* the exact excluded share of the declared target.
  - *W* comes from counts, so it carries no sampling error.
- The benchmark draws its negatives uniformly within O. The within-O weighting term of equation (2) is therefore zero by construction.

**Primary result**
- The exclusion gap G = Δ_target − Δ_O = *W*(Δ_U − Δ_O), with its 95% interval.
- G is the change in the between-model gap caused only by which candidate negatives enter, with models and positives fixed.

**Reported with it, not as further primary results**
- Δ_O, Δ_U and Δ_target.
- Each model's AUROC on O and on U.
- *W*.

**Reading rule**
1. Reversal: Δ_O and Δ_target have opposite signs and both 95% intervals exclude zero.
2. Otherwise, if Δ_O's interval excludes zero, the amplification or shrinkage Δ_target/Δ_O is reported with its interval.
3. If neither applies, no statement about the order is made. G is still reported.

## 7. Intervals

**Sampling uncertainty**
- Protein-level product-weight bootstrap with B = 2,000 replicates.
- In each replicate, every protein receives one Poisson(1) weight, shared wherever it appears:
  - at either end of any pair;
  - in positives and in negatives.
- A pair's weight is the product of its two proteins' weights.
- Both models are evaluated under the same replicate weights.
- Percentile 95% intervals.

**Label uncertainty, reported separately as a band**
- Positives are fixed. Suppose at most a fraction ε_R of region R's sampled candidate negatives are in fact interacting pairs.
- Then each model's AUROC in R moves by at most ε_R.
- Δ_R moves by at most 2ε_R.
- G moves by at most 2*W*(ε_O + ε_U).
- The band is shown for ε ∈ {0.01, 0.02, 0.05}. It is never merged into the bootstrap interval.

## 8. Gates

These are written before any new score is read. All three must pass before the full run.

**G1. Each model is reproduced separately**

*TUnA-R*
- Score a fixed random subset of 300 pairs (seed 20261118) taken from SNOOPPI's released `snooppi_positive__all_models.parquet`.
- Pass: Pearson correlation ≥ 0.999 with the released TUnA-R scores, and maximum absolute difference ≤ 0.01.

*D-SCRIPT*
- Score the full official `human_test.tsv` with `human_v1.sav`.
- Pass: the authors' reported human test-set AUPR is reproduced within 0.02.
- The reference value is read from the original paper and written here before the run. **Not yet recorded:** the paper could not be retrieved on 2026-11-18.
- SNOOPPI did not score D-SCRIPT, so it cannot serve as D-SCRIPT's check.

**G2. Release overlap can be screened**
- Both models' released datasets are obtained in full.
- At least 95% of their pairs resolve to sequences.
- Exact-identity matching against the target runs end to end.
- If TUnA-R's exact training partition cannot be determined, all Bernett partitions are excluded, as specified above.

**G3. Enough distinct proteins remain**
- After filtering, target O spans ≥ 1,000 distinct proteins, and so does target U.
- Positives number ≥ 2,000 pairs over ≥ 1,000 proteins.
- Each sampled region touches ≥ 500 distinct proteins.

**After the gates**
- If a gate fails, the study stops and the failing gate is recorded. No score on the O or U samples is computed.
- If all pass, the full analysis runs once with the fixed seeds and is reported whatever the order.
- A rerun is allowed only to fix a documented bug.

## 9. Secondary analysis (descriptive): UPNA-PPI's released negatives

- Use a uniform sample of 10,000 pairs from UPNA-PPI's released list, restricted in the same way.
- Report both models' AUROC against the same positives. This answers how far the two models differ on the negatives readers actually download.
- No *W* is derived from the released list. Absence from the list shows only that a pair was not released, not that the rule gave it zero probability.
  - The source positive network is proprietary.
  - The list's exhaustiveness over all eligible pairs is unverified.
- Run only after the primary analysis; it needs one further download.

## 10. Deferred

- PLM-interact (GPU).
- UPNA-PPI as a compared model.
- Other networks; STRING ≥ 900 would be the first sensitivity network.

## 11. Manuscript consequences if run

- One Results paragraph (about 150 words).
- One Methods subsection.
- One Extended Data item.
- References: TUnA, D-SCRIPT, the Bernett gold standard, and SNOOPPI if its released scores are used for G1.
- The transfer note's "analyses … unchanged" must change.

## 12. Downloads required

Each download needs Justin's approval. Each goes into its own new, empty directory under `real_model_ppi/downloads/`.

| Item | Source | Size |
|---|---|---|
| TUnA-R weights, revision 4e9ab56 | huggingface.co/yk0/tuna-r-tuna | 135.4 MB |
| TUnA-R code (commit recorded) | github.com/young-su-ko/TUnA-R | small; size recorded at clone time |
| ESM-2 150M weights + contact regression | dl.fbaipublicfiles.com/fair-esm | 592.8 MB + 3 KB |
| D-SCRIPT `human_v1.sav` | cb.csail.mit.edu/cb/dscript | 2.5 MB |
| Bepler–Berger LM `lm_v1.sav` | cb.csail.mit.edu/cb/dscript | 238.3 MB |
| D-SCRIPT `human_train.tsv`, `human_test.tsv`, `human.fasta` | github.com/samsledje/D-SCRIPT | 18.6 + 2.3 + 30.9 MB |
| Bernett partitions + sequences (`data/raw/bernett/`) | github.com/Wang-lab-UCSD/TUnA | 3.8 + 11.6 MB |
| STRING v12.0 human sequences | stringdb-downloads.org | 6.8 MB |
| SNOOPPI released scores: one zip member, `snooppi_positive__all_models.parquet`, read by HTTP range requests; plus `seq_df.csv` (50.7 MB) if the parquet lacks sequences | zenodo.org/records/22116351 | 17.5 MB |
| Python packages into a new Python 3.12 virtual environment: torch (CPU), fair-esm, dscript, TUnA-R's dependencies, pyarrow | PyPI | about 300 MB |

Total about 1.1 GB, or about 1.2 GB with `seq_df.csv`. 365 GB is free on C:.

## Changes from the first proposal (2026-11-18, same day)

- **Withdrawn:** *W* ≈ 0.98 computed from UPNA-PPI's released list.
  - Its numerator and denominator did not align: 178 of the list's 5,037 proteins lie outside the stated universe, and known edges were not removed.
  - Absence from the list does not establish zero sampling probability.
- **Replaced:** the stop rule "both models reproduce SNOOPPI's scores". D-SCRIPT was never scored by SNOOPPI, so that rule would have stopped the study for the wrong reason. G1 now checks each model against its own reference.
- **Corrected:** TUnA-R uses ESM-2 150M, not 650M.
- **Added:** the attribution limit in section 2.

## Amendment 1 (2026-11-18, before any new score was computed)

Justin approved the design, every download in section 12 and the stated choices on 2026-11-18, and asked that computation run on HRP where possible. The items below fill gaps or record facts found while preparing the run. None of them was chosen after seeing a score: at the time of writing no TUnA-R or D-SCRIPT score had been computed on any pair.

**A1. Where each step runs**
- Label-free steps run locally: downloading the small data files, building the target, drawing the samples, and checking G2 and G3. Together they take about 30 seconds and need no GPU.
- Model weights, environments and every scoring step run on HRP, with one GPU per job.

**A2. Environments** (replaces the last row of the section 12 table)
- TUnA-R: the authors' lock file at commit `104756851004a05ff4a27caf1c4c5f4a556cff7d`, installed with `uv sync --frozen --no-dev` under Python 3.13. The lock pins torch 2.7.1 from PyPI (CUDA 12.6 runtime). uv itself comes from PyPI.
- D-SCRIPT: a separate environment, Python 3.12, `dscript==0.3.1` from PyPI, with torch 2.7.1.
- These GPU builds replace the CPU build assumed in section 12, so the package downloads are larger than 300 MB. Versions and hashes are recorded at install time.

**A3. TUnA-R scoring path**
- The path is the one in SNOOPPI's `run_tuna_june.py`:
  - `ESMEmbedder` with one sequence per forward pass, so no padding;
  - `EmbeddingStore`;
  - `Predictor.predict_from_embeddings`, whose returned probability is the score.
- Weights are the snapshot of `yk0/tuna-r-tuna` at revision `4e9ab56ffe2768daf632949543e5f440a4cd5a58`.
- If `pytorch_model.bin` does not have sha256 `6756d40bb36c5f642014abc946200c56b64dd26feea5d0aebb6e515e5885e0bb` (SNOOPPI's pin and the Hugging Face LFS id), the job stops.

**A4. D-SCRIPT scoring path**
- The classifier is `human_v1.sav` (2,542,758 bytes), loaded as the pickled model object.
- The language model comes from the package's `get_pretrained("lm_v1")`. The package's own file for it is `dscript_lm_v1.pt` (238,235,733 bytes), from the same host as `lm_v1.sav`; it is downloaded once and placed where the package looks for it. `lm_v1.sav` is also downloaded and compared tensor by tensor, as a record only, not a gate.
- These are the same calls the command-line `predict_serial` makes.
- **Caching change, to avoid storing about 100 GB of 6,165-dimensional embeddings:**
  - Each protein's language-model output is passed once through the model's own projection (`model.embedding`), and the 100-dimensional result is cached.
  - `map_predict` is then called with `model.embedding` set to `None`. This skips only the projection that has already been applied; every other operation is unchanged.
- **Consistency check (part of G1 for D-SCRIPT).** On 50 pairs of `human_test.tsv` (rows drawn with `default_rng(20261118)`), the wrapper's scores must match the official `dscript predict_serial` run on CPU to within 0.001 (maximum absolute difference).
- If the pickled model cannot be loaded, G1 fails and the failure is recorded. There is no silent fallback to another loading route.
- TF32 is switched off for D-SCRIPT on the GPU, so that the GPU arithmetic matches the CPU reference used by the consistency check. TUnA-R runs with PyTorch's defaults, as SNOOPPI's runs did.

**A4b. Pair orientation**
- Each pair is scored once, in the order written in the sample files; the STRING identifier that sorts first is protein A.
- D-SCRIPT is symmetric by construction, because its contact convolution is forced to be transpose-invariant. TUnA-R is not guaranteed to be, so the fixed order applies equally to positives and to both regions.

**A5. D-SCRIPT G1 reference** (fills the "Not yet recorded" item in section 8)
- The original paper reports human performance only from five-fold cross-validation on its human data set: Table 1 gives AUPR 0.516 and AUROC 0.833. No value has been published for the released `human_v1` model on the released `human_test.tsv` (4,794 positives, 47,931 negatives), and Topsy-Turvy gives none either.
- G1 for D-SCRIPT therefore passes when all of these hold:
  - the consistency check in A4 passes;
  - on the full `human_test.tsv`, AUROC is within 0.05 of 0.833;
  - AUPR, computed as average precision, is within 0.10 of 0.516.
- These tolerances replace the 0.02 first written. The reference is a cross-validation average, not a value for this model on this file. The check is meant to catch a broken pipeline, which gives an AUROC near 0.5.

**A6. Release exclusion**
- TUnA-R's own training files (`data/gold_standard/Intra{0,1,2}_interaction_1500_or_less.tsv` at the commit in A2) are added to the excluded datasets.
- All 231,523 of their pairs also appear among the Bernett partitions that were already excluded.

**A7. Random streams**
- Positives, O and U are drawn from three independent streams spawned from one seed: `numpy.random.SeedSequence(20261118).spawn(3)`, in that order.
- The 300 TUnA-R G1 rows and the 50 D-SCRIPT check pairs each come from `default_rng(20261118)`.
- The bootstrap uses `default_rng(20261118)`.

**A8. Label-free results** (`scripts/build_target.py`; `results/target/target_summary.json`)

| Quantity | Value |
|---|---|
| Scorable proteins | 8,320 of 10,746. 2,405 lie outside 50–800 residues, 21 contain non-standard letters, and none lack a sequence. |
| Released sequence pairs | 743,780. They map to 284,201 STRING protein pairs; 23,555 of those are edges. |
| Declared target *T* | 34,347,139 pairs |
| Target O | 2,733,253 pairs over 2,560 proteins |
| Target U | 31,613,886 pairs over 8,320 proteins |
| ***W* on the declared target** | **0.9204** (unrestricted: 0.9248) |
| Eligible pairs lost | 1,590,094 to scorability; 11,026 to release exclusion |
| Positives | 58,853 scorable edges, 21,133 of them in released data. Sample: 10,000 of the 37,720 that remain, over 4,797 proteins |
| Samples | O: 10,000 pairs over 2,557 proteins. U: 10,000 pairs over 7,526 proteins |

- **G2 passes.** Every pair in each of the 11 released files resolves to sequences.
- **G3 passes** on every count.

Descriptive results, reported with the outcome:
- In 93.0% of target pairs, both proteins appear somewhere in the released data (O 92.5%, U 93.0%).
  - The comparison is therefore mainly on unseen pairs of seen proteins, and the result will say so.
- The share of target pairs with a STRING physical score in [400, 700) is 0.023% in O and 0.155% in U. In the samples it is 0.07% and 0.13%.

**A9. Gating and job order on HRP**
1. The G1 jobs for the two models run first.
2. A gate job then checks both against A3–A5 and the section 8 thresholds.
3. Scoring of the positives and of the O and U samples starts only if the gate job succeeds, through a Slurm `afterok` dependency. The scoring jobs also check for the gate's pass file before they load any study pair.
4. Scores are not examined until both models' study scores have been fetched.

**A10. Weighted AUROC used by the bootstrap**
- In each replicate, every protein in the union of the samples receives one Poisson(1) weight. A pair's weight is the product of its two proteins' weights.
- AUROC_w = Σ_i Σ_j w_i w_j [1(s_i > s_j) + ½·1(s_i = s_j)] / (Σ_i w_i · Σ_j w_j), with i over positives and j over the region's negatives.
- *W* is held at 0.9204 in every replicate.
- A replicate in which a region has zero total positive or negative weight is redrawn, and the number of redraws is reported.

## Amendment 2 (2026-11-18, before any G1 or study score was computed)

These facts were found in the smoke test on HRP. That test scores only six synthetic toy sequences and never reads a STRING, `human_test.tsv` or SNOOPPI pair. No G1 or study score had been computed when this was written.

**A11. Route of the official CLI check** (A4)
- In `dscript==0.3.1` from PyPI:
  - `-d/--device` takes an integer, with −1 meaning CPU, so `--device cpu` is rejected.
  - The one-step route `predict_serial --seqs` stops with a `TypeError`: it calls `parse(path, "r")`, but `parse` takes one argument.
- The check therefore uses the package's two-step route:
  1. `dscript embed --seqs SUBSET.fasta -o SUBSET.h5 --device=-1`
  2. `dscript predict_serial --pairs SUBSET.tsv --embeddings SUBSET.h5 --model human_v1.sav --device=-1 --load_proc 1`
- Both are the package's own commands. Step 2 makes the same `map_predict` call on the full language-model embeddings, so it remains an independent check of the wrapper's cached projection.
- The 50 pairs, the CPU device and the 0.001 tolerance are unchanged.
- The h5 file stays in the work directory on HRP and is not fetched.

**A12. The record-only comparison in A4**
- Comparing `lm_v1.sav` with the package's language model tensor by tensor was not possible. Loading the pickled `lm_v1.sav` under torch 2.7.1 fails with `AttributeError: 'LSTM' object has no attribute '_flat_weights'`.
- It is recorded as "comparison not possible". It was never a gate, so nothing else changes.
- The scoring path in A4 does not use `lm_v1.sav`.

**A13. How the gate in A9 is enforced**
- A9 item 3 named a Slurm `afterok` dependency and a pass-file check inside the scoring jobs. As implemented, all stages run in a single Slurm job (`scripts/hpc/run_all.sbatch`), in order, under `set -e`:
  1. G1 for TUnA-R;
  2. G1 for D-SCRIPT;
  3. the CLI check;
  4. the gate script, which exits with an error if any check fails;
  5. a shell test for the gate's pass file;
  6. scoring of the study pairs.
- The scoring scripts do not check for the pass file themselves.
- The effect A9 requires is unchanged: no study pair is scored unless the gate has passed.

## Amendment 3 (2026-11-18, after the G1 results were seen, before any study score was computed)

**A14. G1 results** (job 4598207 on mantis-027, an NVIDIA A100 80GB, 10:16–10:29 local time)
- TUnA-R passed: 300 of 300 rows, Pearson r = 0.99999999999919, maximum absolute difference 7.2 × 10⁻⁷.
- D-SCRIPT, on the full `human_test.tsv` (52,725 pairs, 4,794 positives, all scored):
  - AUROC 0.8539, inside 0.833 ± 0.05: pass.
  - Average precision 0.6173, outside 0.516 ± 0.10. It is above the upper limit (0.616) by 0.0013: fail.
  - Official CLI check passed: 50 of 50 pairs, maximum absolute difference 5.7 × 10⁻⁶ (tolerance 0.001).
- The gate therefore failed (`out/gate/gate.json`). The job stopped before any study pair was scored, as A13 intends.

**A15. Checks after the failure**
- AUROC and average precision were recomputed locally from the fetched scores, with code that does not use scikit-learn. Results: AUROC 0.8539, average precision 0.6173, trapezoidal AUPR 0.6172.
- The reference was re-read from the paper (PMC8586911, Table 1, "H. sapiens (5-fold cross validation)"): D-SCRIPT AUPR 0.516, AUROC 0.833. It had been transcribed correctly.
- The failure is therefore not a bug. The released model scores higher on its own test file than the paper's cross-validation average.
- One possible reason, not tested: the paper set aside 20% of its human pairs (9,587 positives) for validation. The released `human_test.tsv` holds 4,794 positives, and the paper does not say whether the file was used to choose the released checkpoint.

**A16. Amended G1 criterion for D-SCRIPT** (decided by Justin on 2026-11-18)
- A5 states that the check is meant to catch a broken pipeline. A broken pipeline lowers performance, so a two-sided tolerance did not match that purpose.
- The criterion becomes one-sided:
  - AUROC ≥ 0.783 (0.833 − 0.05);
  - average precision ≥ 0.416 (0.516 − 0.10).
- Unchanged: the CLI check (≤ 0.001 on all 50 pairs) and the TUnA-R criterion. AUROC passes under either version.
- **This is a deviation from section 8**, which says the study stops when a gate fails.
  - It was made after the G1 results were seen, but before any study pair was scored.
  - G1 uses no study pair, so the change cannot depend on Δ_O, Δ_U or G.
  - The manuscript's Methods will report it as a deviation, with the numbers in A14.

**A17. How the run continues**
- G1 is not rerun. Its outputs and the failing `out/gate/gate.json` are kept unchanged.
- A new script, `scripts/hpc/gate_check_a3.py`, applies A16 to the same G1 files and writes `out/gate_a3/`.
  - Run locally on the fetched G1 outputs, the unchanged `gate_check.py` reproduces the failure with the numbers above, and `gate_check_a3.py` passes.
- A new job, `scripts/hpc/run_main.sbatch`, runs `gate_check_a3.py` and tests for `out/gate_a3/PASS`. Only then does it score the positives and the O and U samples, once.
- Samples, seeds, the analysis and the reading rule are unchanged.

## Amendment 4 (2026-11-18, after the primary result was known, before any Topsy-Turvy score was computed)

**A18. A third model, added after the primary result**
- Justin decided on 2026-11-18 to add one model. The primary result (TUnA-R vs D-SCRIPT, job 4610895, `results/analysis_4610895.json`) was already known.
- The extension is secondary. It does not change the primary result or its reading.
- The model is Topsy-Turvy (Singh et al., *Bioinformatics* 2022). It was chosen for three reasons, all fixed before any of its scores was computed:
  - It is the default pretrained model of the dscript package: `get_pretrained()` defaults to `human_v2`, documented as the "Human trained model from Topsy-Turvy manuscript".
  - Its paper describes the same human data set and the same 80/20 split as D-SCRIPT: STRING v11, 38,345 training positives. The released D-SCRIPT training and test files are already excluded (A6), so the declared target, *W* and the three samples stay unchanged.
    - If Topsy-Turvy's training pairs differed from the released files, they could not be excluded, as for any unreleased data.
  - It uses D-SCRIPT's architecture and environment, so the A4 scoring path applies unchanged.
- Known limitation: the paper tuned two hyperparameters by cross-validation on the whole human network, so that tuning may have seen target pairs. This is not training on them, but it is reported.
- No further model will be added to this test.

**A19. Weights and G1 data** (downloaded on 2026-11-18 with Justin's approval)
- `dscript_human_v2.pt`:
  - 2,539,686 bytes, sha256 `6b217a1b0399f6a0d72e4f4956d6f87bcf49aefb8ce37ed3636c8fe8a4731bf4`, from the authors' host (`https://cb.csail.mit.edu/dscript/data/models/`).
  - It is the file `dscript.pretrained.get_pretrained("human_v2")` loads. It is a CPU-stored state dict and is built with the package's own `build_human_1`.
  - The package README now recommends a Hugging Face copy (`samsl/topsy_turvy_human_v1`). That copy was not downloaded or compared.
- From the D-SCRIPT repository at the commit in A2 (`23cbb0f…`):
  - `data/pairs/mouse_test.tsv`: 2,970,000 bytes, sha256 `cbea2862659189054620fedb68cfac51271754843fd85e20e5ae98b1a8bb7296`. It holds 55,000 pairs (5,000 positives) over 37,497 proteins, all 50–800 residues long.
  - `data/seqs/mouse.fasta`: 17,029,641 bytes, sha256 `17a22b563760987f5ba4732e55b02327eb2862b65e9685c0207e190bd4092140`.
  - Both are used only for G1.
- HRP downloads the same three files again, and they must match these hashes.

**A20. Scoring path**
- Scoring follows A4 for D-SCRIPT, with the Topsy-Turvy weights:
  - the package language model `lm_v1`, set up as in `lm_embed`;
  - the cached projection through the model's own `embedding`, then `map_predict` with `embedding = None`;
  - TF32 off;
  - pairs in the order of the sample files (A4b).
- The CLI check needs a model file that `predict_serial` can load. `predict_serial` reads a `.sav`/`.pt` argument as a whole pickled model, so the model built by `build_human_1` is saved whole with `torch.save` and passed as `--model`.
  - The check therefore tests the scoring path, not the source of the weights.
  - It otherwise follows A11 (`dscript embed`, then `predict_serial --embeddings`, `--device=-1`).

**A21. G1 for Topsy-Turvy**
- The Topsy-Turvy paper reports no human held-out result; all of its results are cross-species. The reference is its Table 3, *M. musculus*: AUPR 0.735 ± 0.03 and AUROC 0.934 ± 0.01 (mean ± s.d. over three random initializations).
- G1 passes when all of these hold. The limits are one-sided from the start, as in A16.
  - All 55,000 pairs of `mouse_test.tsv` are scored with finite values.
  - AUROC ≥ 0.884 (0.934 − 0.05).
  - Average precision ≥ 0.635 (0.735 − 0.10).
  - On 50 pairs of `mouse_test.tsv`, the wrapper and the official CLI differ by at most 0.001 (maximum absolute difference). The rows are drawn with `default_rng(20261118)`, as for D-SCRIPT, and include 7 positives.
- If G1 fails, the extension stops and the failure is reported. No Topsy-Turvy score is computed on the study samples.

**A22. Secondary estimands and reading**
- There are two secondary pairs, each with Δ = AUROC_Topsy-Turvy − AUROC_other:
  - Topsy-Turvy vs TUnA-R;
  - Topsy-Turvy vs D-SCRIPT.
- For each pair the following are reported:
  - Δ_O, Δ_U, Δ_target and G;
  - each model's AUROC on O, on U and on the target;
  - the section 6 reading rule, applied unchanged.
- Intervals come from the bootstrap in section 7 and A10, with the same seed, protein list and replicates as the primary analysis (`scripts/analyze_tt.py`, which reuses the functions of `scripts/analyze.py`).
  - The primary pair is recomputed in the same run and must match `results/analysis_4610895.json` exactly. Otherwise the extension's numbers are not reported until the difference is explained.
  - Tested beforehand with D-SCRIPT's scores standing in for Topsy-Turvy: the primary pair reproduced exactly.
- The ε band is reported as for the primary result.
- These are secondary results:
  - No multiplicity correction is applied.
  - None of them is described as confirming or refuting the primary result.
  - All are reported whatever they show.
  - The manuscript will state that Topsy-Turvy was added after the primary result was known.

**A23. Where it runs**
- A connection to HRP downloads the three files into new directories and checks their hashes. It then runs a load test and the synthetic-sequence smoke test, and submits one job, `scripts/hpc/run_tt.sbatch`. The job runs:
  1. G1 on `mouse_test.tsv`;
  2. the CLI check;
  3. the gate script (`gate_check_tt.py`), followed by a test for `out/gate_tt/PASS`;
  4. scoring of the study samples.
- Outputs go to `out/g1_tt`, `out/gate_tt` and `out/main_tt`. Earlier outputs are not touched.
