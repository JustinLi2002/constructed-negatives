#!/usr/bin/env python3
"""The unobserved region, measured on real tasks rather than bounded.

The worst-case criterion uses only |Delta_bar_U| <= 1.  That is true and
uninformative: it makes the required gap a monotone transform of W, and invites
the objection that any missing data is unidentified under a sufficiently
adversarial assumption.

For the PTM tasks the assumption is unnecessary.  The region a threshold-T
benchmark never samples is unobserved *to that benchmark*, not to us: the first
paper rebuilt the same tasks without the threshold, so the predictions it
produced cover both sides of T and Delta_k can be measured on both.

What must be measured is the right quantity.  The decomposition requires

    Delta_k = AUROC(all positives, stratum-k negatives)_aug
            - AUROC(all positives, stratum-k negatives)_base

with the positive set held FIXED and only the negatives stratified -- that is
what makes AUROC linear in the negative distribution.  It is not the within-bin
AUROC difference reported by the first paper's depth_stratified.py, which scores
a bin's positives against the same bin's negatives.  That paper says explicitly
that its within-bin measure is expected to be weak, because the shortcut acts
between proteins and binning by depth removes the very effect it is meant to
capture.  Using that column here gives c of order 0.005 and a decomposition that
fails to reconcile with the paper's own headline numbers by a factor of four,
which is how the error was caught.

With c measured, |Delta_bar_U - Delta_O| <= c gives
Delta_deploy >= Delta_O - W c, so an ordering with observed gap D_O is robust
whenever D_O > W c -- a requirement that carries a property of the task rather
than only the sampling rate.

    python measured_c.py --root ~/HRP/pdisjoint_runs_v2 --data ~/HRP/rebuilt
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PATTERN = (r"^(?P<task>[a-z_0-9]+?)__(?P<train>replica|rebuilt)"
           r"__(?P<cond>baseline|ppi|shuffled)__split(?P<split>\d+)"
           r"(?:__(?P<source>esm|prott5))?"
           r"__on_(?P<eval>replica|rebuilt)\.pred\.tsv$")
PROTEIN, SITE, LABEL, SCORE = "protein", "pos", "y", "y_pred"
ORDER = ["phosphorylation_st", "phosphorylation_y", "acetylation_k",
         "methylation_k", "methylation_r", "sumoylation_k",
         "ubiquitination_k", "glycosylation_n"]
W_DEPLOY = {"phosphorylation_st": 0.605, "phosphorylation_y": 0.642,
            "acetylation_k": 0.921, "methylation_k": 0.974,
            "methylation_r": 0.977, "sumoylation_k": 0.913,
            "ubiquitination_k": 0.750, "glycosylation_n": 0.977}
T = 10


def family(cond, source):
    if cond == "baseline":
        return "baseline"
    return {None: "interaction", "esm": "esm2", "prott5": "prott5"}[source]


def depth_of(data_dir, task):
    df = pd.read_csv(os.path.join(data_dir, "%s_all.tsv" % task), sep="\t")
    cols = {c.lower(): c for c in df.columns}
    p, y = cols.get("protein"), cols.get("y") or cols.get("label")
    return df.groupby(df[p])[y].sum()


def main(root, data, fam, train, ev):
    files = {}
    for path in sorted(glob.glob(os.path.join(root, "*.pred.tsv"))):
        m = re.match(PATTERN, os.path.basename(path))
        if not m:
            continue
        d = m.groupdict()
        if d["train"] == train and d["eval"] == ev:
            files.setdefault((d["task"], family(d["cond"], d["source"])),
                             {})[d["split"]] = path

    tasks = [t for t in ORDER if (t, fam) in files and (t, "baseline") in files]
    if not tasks:
        sys.exit("no task has both baseline and %s for train=%s eval=%s"
                 % (fam, train, ev))

    print("Delta_k with the positive set held fixed, split at depth T = %d" % T)
    print("family=%s train=%s eval=%s\n" % (fam, train, ev))
    print("%-22s %9s %11s %8s %8s %10s %10s"
          % ("task", "Delta_O", "Delta_bar_U", "c", "W", "gap meas.",
             "gap worst"))
    print("-" * 84)

    out = []
    for task in tasks:
        dep = depth_of(data, task)
        frames = []
        for s in sorted(files[(task, fam)]):
            if s not in files[(task, "baseline")]:
                continue
            b = pd.read_csv(files[(task, "baseline")][s], sep="\t")
            a = pd.read_csv(files[(task, fam)][s], sep="\t")
            m = b.merge(a, on=[PROTEIN, SITE, LABEL],
                        suffixes=("_base", "_aug"), validate="one_to_one")
            frames.append(m)
        if not frames:
            continue
        m = pd.concat(frames, ignore_index=True)
        m["depth"] = m[PROTEIN].map(dep)
        m = m.dropna(subset=["depth"])

        pos = m[m[LABEL] == 1]
        neg = m[m[LABEL] == 0]
        if len(pos) < 100 or len(neg) < 100:
            continue

        def delta(sub):
            if len(sub) < 50:
                return np.nan
            y = np.r_[np.ones(len(pos)), np.zeros(len(sub))]
            sb = np.r_[pos[SCORE + "_base"], sub[SCORE + "_base"]]
            sa = np.r_[pos[SCORE + "_aug"], sub[SCORE + "_aug"]]
            return roc_auc_score(y, sa) - roc_auc_score(y, sb)

        d_O = delta(neg[neg.depth >= T])
        d_U = delta(neg[neg.depth < T])
        d_all = delta(neg)
        if not (np.isfinite(d_O) and np.isfinite(d_U)):
            print("%-22s %9s %11s %8s %8s %10s %10s"
                  % (task, "--", "--", "--", "--", "--", "--"))
            continue
        c = abs(d_U - d_O)
        w = W_DEPLOY.get(task, np.nan)
        meas = w * c
        worst = w / (1 - w)
        out.append((task, d_O, d_U, c, w, meas, worst, d_all,
                    (neg.depth < T).mean()))
        print("%-22s %+9.4f %+11.4f %8.4f %8.3f %10.4f %10.2f"
              % (task, d_O, d_U, c, w, meas, worst))
    print("-" * 84)
    if out:
        cs = [r[3] for r in out]
        print("c ranges %.4f to %.4f (median %.4f) against the worst case of 2"
              % (min(cs), max(cs), sorted(cs)[len(cs) // 2]))
        print("\nempirical check of the decomposition, using the negative share")
        print("below T in these test partitions rather than the release's W:")
        print("%-22s %10s %10s %10s %10s"
              % ("task", "Delta_all", "reconstructed", "residual", "share<T"))
        print("-" * 66)
        for t, dO, dU, c, w, meas, worst, d_all, share in out:
            rec = (1 - share) * dO + share * dU
            print("%-22s %+10.4f %+13.4f %10.2e %10.3f"
                  % (t, d_all, rec, abs(d_all - rec), share))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/HRP/pdisjoint_runs_v2"))
    ap.add_argument("--data", default=os.path.expanduser("~/HRP/rebuilt"))
    ap.add_argument("--family", default="esm2")
    ap.add_argument("--train", default="replica")
    ap.add_argument("--eval", default="rebuilt")
    a = ap.parse_args()
    main(a.root, a.data, a.family, a.train, a.eval)
