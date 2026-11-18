"""Check the synthetic-sequence smoke run: every output exists with the right
number of rows, every score is finite, and the D-SCRIPT wrapper matches the
official CLI (both on CPU here) to within 1e-4.  Exit 1 on any failure.

    python -I smoke_check.py SMOKE_DIR
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


t = rows(os.path.join(SM, "g1_tuna", "g1_tuna_scores.tsv"))
finite([r[1] for r in t], "g1_tuna")
d = rows(os.path.join(SM, "g1_dscript", "human_test_scores.tsv"))
finite([r[3] for r in d], "g1_dscript")
wrap = {(r[0], r[1]): float(r[3]) for r in d}
cli = {(r[0], r[1]): float(r[2]) for r in rows(os.path.join(SM, "g1_dscript", "cli_check.tsv"), header=False)}
diffs = {k: abs(cli[k] - wrap[k]) for k in cli if k in wrap}
if len(diffs) != 2:
    problems.append(f"CLI compared {len(diffs)} pairs, expected 2")
for k, v in diffs.items():
    print(f"wrapper vs CLI {k}: wrapper {wrap[k]:.8f} cli {cli[k]:.8f} diff {v:.2e}")
    if v > 1e-4:
        problems.append(f"wrapper and CLI differ by {v:.2e} on {k}")
for model in ("tuna", "dscript"):
    for lab in ("positives", "O", "U"):
        r = rows(os.path.join(SM, f"main_{model}", f"scores_{model}_{lab}.tsv"))
        if len(r) != 2:
            problems.append(f"main_{model} {lab}: {len(r)} rows, expected 2")
        finite([x[2] for x in r], f"main_{model} {lab}")
print(f"g1_tuna rows {len(t)}, g1_dscript rows {len(d)}")
if problems:
    print("SMOKE FAILED:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("SMOKE OK")
