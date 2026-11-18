"""Gate G1 for Topsy-Turvy (DESIGN.md, Amendment 4, A21).

Reads only the Topsy-Turvy G1 outputs.  Writes out/gate_tt/gate.json, and
out/gate_tt/PASS when every check passes; exits 1 otherwise, which stops the job
before any study pair is scored with Topsy-Turvy.

Reference: Topsy-Turvy paper, Table 3, M. musculus: AUPR 0.735 +/- 0.03,
AUROC 0.934 +/- 0.01 (three random initializations).  One-sided lower limits.

    python -I gate_check_tt.py R
"""
import json
import math
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = sys.argv[1]
OUT = os.path.join(R, "out", "gate_tt")
os.makedirs(OUT, exist_ok=False)
N_PAIRS, AUROC_REF, AP_REF = 55000, 0.934, 0.735
AUROC_MIN, AP_MIN = AUROC_REF - 0.05, AP_REF - 0.10

lab, prob, wrap = [], [], {}
with open(os.path.join(R, "out", "g1_tt", "mouse_test_scores.tsv")) as fh:
    fh.readline()
    for line in fh:
        a, b, l, p = line.rstrip("\n").split("\t")
        lab.append(int(float(l)))          # mouse_test.tsv writes labels as 1.0 / 0.0
        prob.append(float(p))
        wrap[(a, b)] = float(p)
lab, prob = np.array(lab), np.array(prob)
complete = len(lab) == N_PAIRS and np.isfinite(prob).all()
auroc = float(roc_auc_score(lab, prob)) if complete else math.nan
ap = float(average_precision_score(lab, prob)) if complete else math.nan

cli = {}
cli_path = os.path.join(R, "out", "g1_tt", "cli_check.tsv")
if os.path.exists(cli_path):
    for line in open(cli_path):
        f = line.split()
        if len(f) >= 3:
            cli[(f[0], f[1])] = float(f[2])
subset = [tuple(line.split()[:2]) for line in open(os.path.join(R, "inputs", "g1_tt_cli_subset.tsv"))
          if line.strip()]
diffs = [abs(cli[p] - wrap[p]) for p in subset if p in cli and p in wrap]
cli_ok = len(diffs) == len(subset) == 50
cli_max = max(diffs) if diffs else math.nan
gate = {"amendment": "DESIGN.md Amendment 4 (A21)",
        "tt": {"pairs_scored": int(len(lab)), "positives": int(lab.sum()), "complete": bool(complete),
               "auroc": auroc, "auroc_reference": AUROC_REF, "auroc_min": AUROC_MIN,
               "average_precision": ap, "aupr_reference": AP_REF, "ap_min": AP_MIN,
               "cli_pairs_compared": len(diffs), "cli_max_abs_diff": cli_max, "cli_tolerance": 1e-3,
               "pass": bool(complete and cli_ok and auroc >= AUROC_MIN and ap >= AP_MIN
                            and cli_max <= 1e-3)}}
gate["pass"] = gate["tt"]["pass"]
with open(os.path.join(OUT, "gate.json"), "w") as out:
    json.dump(gate, out, indent=1)
print(json.dumps(gate, indent=1))
if gate["pass"]:
    open(os.path.join(OUT, "PASS"), "w").close()
    sys.exit(0)
sys.exit(1)
