"""Gate G1 with the amended D-SCRIPT criterion (DESIGN.md, Amendment 3, A16).

Same inputs and computations as gate_check.py, which stays unchanged and whose
failing out/gate/gate.json is kept.  The only change: D-SCRIPT's AUROC and average
precision need only reach the lower limits (0.833 - 0.05 and 0.516 - 0.10).  The
CLI check and the TUnA-R criterion are unchanged.  Writes out/gate_a3/gate.json,
and out/gate_a3/PASS when every check passes; exits 1 otherwise.

    python -I gate_check_a3.py R
"""
import json
import math
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = sys.argv[1]
OUT = os.path.join(R, "out", "gate_a3")
os.makedirs(OUT, exist_ok=False)
gate = {"amendment": "DESIGN.md Amendment 3 (A16): one-sided lower limits for D-SCRIPT AUROC and AP"}

# ---- TUnA-R against SNOOPPI's released scores (unchanged) --------------------
ref = {}
with open(os.path.join(R, "inputs", "g1_tuna_pairs.tsv")) as fh:
    fh.readline()
    for line in fh:
        f = line.rstrip("\n").split("\t")
        ref[int(f[0])] = float(f[3])
new = {}
with open(os.path.join(R, "out", "g1_tuna", "g1_tuna_scores.tsv")) as fh:
    fh.readline()
    for line in fh:
        f = line.rstrip("\n").split("\t")
        new[int(f[0])] = float(f[1])
keys = sorted(ref)
x = np.array([ref[k] for k in keys])
y = np.array([new.get(k, math.nan) for k in keys])
ok_rows = len(keys) == 300 and set(new) == set(ref) and np.isfinite(y).all()
r = float(np.corrcoef(x, y)[0, 1]) if ok_rows else math.nan
d = float(np.max(np.abs(x - y))) if ok_rows else math.nan
gate["tuna"] = {"pairs": len(keys), "pearson": r, "max_abs_diff": d,
                "mean_abs_diff": float(np.mean(np.abs(x - y))) if ok_rows else math.nan,
                "pass": bool(ok_rows and r >= 0.999 and d <= 0.01)}

# ---- D-SCRIPT on the official human test file ------------------------------
lab, prob, wrap = [], [], {}
with open(os.path.join(R, "out", "g1_dscript", "human_test_scores.tsv")) as fh:
    fh.readline()
    for line in fh:
        a, b, l, p = line.rstrip("\n").split("\t")
        lab.append(int(l))
        prob.append(float(p))
        wrap[(a, b)] = float(p)
lab, prob = np.array(lab), np.array(prob)
complete = len(lab) == 52725 and np.isfinite(prob).all()
auroc = float(roc_auc_score(lab, prob)) if complete else math.nan
ap = float(average_precision_score(lab, prob)) if complete else math.nan

cli = {}
cli_path = os.path.join(R, "out", "g1_dscript", "cli_check.tsv")
if os.path.exists(cli_path):
    for line in open(cli_path):
        f = line.split()
        if len(f) >= 3:
            cli[(f[0], f[1])] = float(f[2])
subset = [tuple(line.split()[:2]) for line in open(os.path.join(R, "inputs", "g1_dscript_cli_subset.tsv"))
          if line.strip()]
diffs = [abs(cli[p] - wrap[p]) for p in subset if p in cli and p in wrap]
cli_ok = len(diffs) == len(subset) == 50
cli_max = max(diffs) if diffs else math.nan
AUROC_MIN, AP_MIN = 0.833 - 0.05, 0.516 - 0.10
gate["dscript"] = {
    "pairs_scored": int(len(lab)), "positives": int(lab.sum()), "complete": bool(complete),
    "auroc": auroc, "auroc_min": AUROC_MIN, "average_precision": ap, "ap_min": AP_MIN,
    "cli_pairs_compared": len(diffs), "cli_max_abs_diff": cli_max, "cli_tolerance": 1e-3,
    "pass": bool(complete and cli_ok and auroc >= AUROC_MIN and ap >= AP_MIN and cli_max <= 1e-3)}

gate["pass"] = gate["tuna"]["pass"] and gate["dscript"]["pass"]
with open(os.path.join(OUT, "gate.json"), "w") as out:
    json.dump(gate, out, indent=1)
print(json.dumps(gate, indent=1))
if gate["pass"]:
    open(os.path.join(OUT, "PASS"), "w").close()
    sys.exit(0)
sys.exit(1)
