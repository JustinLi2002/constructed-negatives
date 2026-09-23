#!/usr/bin/env python3
"""Which parts of this argument depend on AUROC, and which do not.

Three claims in this work have different scopes and the difference matters,
because much of the PTM literature reports AUPRC rather than AUROC and a
reviewer will ask.

  The violation itself does not depend on any metric.  Positivity is a property
  of the sampling design: entities below the threshold have propensity zero, no
  reweighting estimator is defined there, and that is true whatever is being
  estimated.

  The exact decomposition does depend on the metric.  AUROC is a mean over
  positive-negative pairs, so it is linear in the negative distribution and
  Delta_deploy = sum_k w_k Delta_k holds exactly.  AUPRC is not: precision at
  any operating point depends on the ratio of positives to negatives, which
  changes when strata of different size are mixed.

  Everything built on the decomposition therefore inherits its scope.  The
  measurement of c, and with it the ability to say how wrong a benchmark is
  rather than merely that it is unidentified, is available for AUROC and not for
  AUPRC.

This script measures the gap rather than asserting it: for each task it computes
the per-stratum values with the positive set held fixed, forms sum_k w_k m_k,
and compares with the value on all negatives together.

    python metric_linearity.py --root ~/HRP/pdisjoint_runs_v2 --data ~/HRP/rebuilt
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from measured_c import (ORDER, PATTERN, PROTEIN, SCORE, SITE, LABEL, depth_of,
                        family)
from prediction_files import index_predictions

BINS = [0, 1, 2, 3, 5, 10, 20, 50, np.inf]


def collect(root, data, fam, train, ev):
    return {(task, model): splits for (task, model, evaluation), splits
            in index_predictions(root, train).items() if evaluation == ev}


def main(root, data, fam, train, ev):
    files = collect(root, data, fam, train, ev)
    tasks = [t for t in ORDER if (t, fam) in files and (t, "baseline") in files]
    if not tasks:
        sys.exit("no usable task")

    print("Does the stratum decomposition hold for this metric?")
    print("sum_k w_k m_k against m computed on all negatives together,")
    print("positive set held fixed, %d depth strata.\n" % (len(BINS) - 1))
    print("%-22s %12s %12s %12s %12s"
          % ("task", "AUROC resid", "AUPRC resid", "AUPRC value", "resid/value"))
    print("-" * 74)

    rows = []
    for task in tasks:
        dep = depth_of(data, task)
        frames = []
        for s in sorted(files[(task, fam)]):
            if s not in files[(task, "baseline")]:
                continue
            b = pd.read_csv(files[(task, "baseline")][s], sep="\t")
            a = pd.read_csv(files[(task, fam)][s], sep="\t")
            frames.append(b.merge(a, on=[PROTEIN, SITE, LABEL],
                                  suffixes=("_base", "_aug"),
                                  validate="one_to_one"))
        if not frames:
            continue
        m = pd.concat(frames, ignore_index=True)
        m["depth"] = m[PROTEIN].map(dep)
        m = m.dropna(subset=["depth"])
        pos, neg = m[m[LABEL] == 1], m[m[LABEL] == 0]
        if len(pos) < 100 or len(neg) < 100:
            continue
        b = np.digitize(neg.depth.values, BINS[1:-1], right=False)

        def metric(fn, sub):
            y = np.r_[np.ones(len(pos)), np.zeros(len(sub))]
            return fn(y, np.r_[pos[SCORE + "_base"], sub[SCORE + "_base"]])

        for name, fn in (("roc", roc_auc_score), ("prc", average_precision_score)):
            whole = metric(fn, neg)
            acc = 0.0
            for k in range(len(BINS) - 1):
                sub = neg[b == k]
                if len(sub) == 0:
                    continue      # every non-empty stratum, or the weights
                                  # do not sum to one and the residual picks
                                  # up the dropped mass rather than the metric
                acc += (len(sub) / len(neg)) * metric(fn, sub)
            if name == "roc":
                r_roc, v_roc = abs(acc - whole), whole
            else:
                r_prc, v_prc = abs(acc - whole), whole
        rows.append((task, r_roc, r_prc, v_prc))
        print("%-22s %12.2e %12.2e %12.4f %11.1f%%"
              % (task, r_roc, r_prc, v_prc, 100 * r_prc / max(v_prc, 1e-12)))
    print("-" * 74)
    if rows:
        print("AUROC residual max %.1e -- the decomposition is exact."
              % max(r[1] for r in rows))
        print("AUPRC residual %.3f to %.3f, %.0f%% to %.0f%% of the value it is"
              % (min(r[2] for r in rows), max(r[2] for r in rows),
                 100 * min(r[2] / r[3] for r in rows),
                 100 * max(r[2] / r[3] for r in rows)))
        print("meant to reconstruct -- the decomposition does not hold, and")
        print("nothing built on it transfers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/HRP/pdisjoint_runs_v2"))
    ap.add_argument("--data", default=os.path.expanduser("~/HRP/rebuilt"))
    ap.add_argument("--family", default="esm2")
    ap.add_argument("--train", default="replica")
    ap.add_argument("--eval", default="rebuilt")
    a = ap.parse_args()
    main(a.root, a.data, a.family, a.train, a.eval)
