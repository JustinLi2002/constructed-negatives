"""Check the synthetic-sequence smoke run for Topsy-Turvy: every output exists with
the right number of rows, every score is finite, and the wrapper matches the
official CLI (both on CPU here) to within 1e-4.  Exit 1 on any failure.

    python -I smoke_check_tt.py SMOKE_DIR
"""
import math
import os
import sys

SM = sys.argv[1]
problems = []


def rows(path, header=True):
    if not os.path.exists(path):
        problems.append(f"missing {path}")
        return []
    with open(path) as fh:
        lines = [line.rstrip("\n").split("\t") for line in fh if line.strip()]
    return lines[1:] if header else lines


def finite(vals, label):
    for v in vals:
        if not math.isfinite(float(v)):
            problems.append(f"non-finite score in {label}")
            return


d = rows(os.path.join(SM, "g1_tt", "human_test_scores.tsv"))
finite([r[3] for r in d], "g1_tt")
if len(d) != 3:
    problems.append(f"g1_tt: {len(d)} rows, expected 3")
wrap = {(r[0], r[1]): float(r[3]) for r in d}
cli = {(r[0], r[1]): float(r[2]) for r in rows(os.path.join(SM, "g1_tt", "cli_check.tsv"), header=False)}
diffs = {k: abs(cli[k] - wrap[k]) for k in cli if k in wrap}
if len(diffs) != 2:
    problems.append(f"CLI compared {len(diffs)} pairs, expected 2")
for k, v in diffs.items():
    print(f"wrapper vs CLI {k}: wrapper {wrap[k]:.8f} cli {cli[k]:.8f} diff {v:.2e}")
    if v > 1e-4:
        problems.append(f"wrapper and CLI differ by {v:.2e} on {k}")
for lab in ("positives", "O", "U"):
    r = rows(os.path.join(SM, "main_tt", f"scores_tt_{lab}.tsv"))
    if len(r) != 2:
        problems.append(f"main_tt {lab}: {len(r)} rows, expected 2")
    finite([x[2] for x in r], f"main_tt {lab}")
print(f"g1_tt rows {len(d)}")
if problems:
    print("SMOKE FAILED:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("SMOKE OK")
