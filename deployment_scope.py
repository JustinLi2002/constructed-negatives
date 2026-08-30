#!/usr/bin/env python3
"""The deployment population a benchmark's comparisons are actually about.

W is not a property of a benchmark alone.  It is the share of *some* deployment
population's negatives that the benchmark never samples, and that population is
a choice the analyst makes.  Reporting W against the whole reference proteome
invites the reply that the benchmark's authors never claimed to be deployed
there, and answering with a second, more conservative universe is a hedge rather
than an answer.

The better move is to invert the question.  Rather than asking how bad W is on a
population we picked, ask which populations the benchmark could support:

    what is the smallest d_min such that, restricted to entities carrying at
    least d_min positives, the guarantee becomes available?

Two readings.  Without a reconstruction the binding condition is W <= 0.5, since
|delta_bar_U - delta_O| can be as large as 1 + delta_O.  With one it is
W <= delta_O / c for the measured c, which is weaker and sometimes vacuous -- a
task whose unsampled region behaves exactly as its sampled one imposes no
restriction at all.  The worst-case column is what a reader of a published
benchmark can compute; the measured column is what is true.  The answer converts "your rankings do not
hold" into "your rankings hold only over this population", which is the same
force, is not open to the charge of estimand-shopping, and is something a
benchmark builder can act on -- it is the scope statement their paper should
have carried.

    PTM_AUDIT_BASE=/home/FCAM/juli/HRP python deployment_scope.py
"""
import os
from collections import defaultdict
from itertools import groupby

import numpy as np
import pandas as pd

BASE = os.environ.get("PTM_AUDIT_BASE", "/home/FCAM/juli/HRP")
FASTA = f"{BASE}/deepmvp/DeepMVP/data/swiss_prot_human_20190214.fasta"
REBUILT = f"{BASE}/rebuilt"
T = 10

TARGETS = {"acetylation_k": "K", "glycosylation_n": "N", "methylation_k": "KR",
           "methylation_r": "R", "phosphorylation_st": "ST",
           "phosphorylation_y": "Y", "sumoylation_k": "K",
           "ubiquitination_k": "K"}
MERGED = {"acetylation_k": ["acetylation_k"],
          "glycosylation_n": ["glycosylation_n"],
          "methylation_k": ["methylation_k"],
          "methylation_r": ["methylation_k"],
          "phosphorylation_st": ["phosphorylation_st", "phosphorylation_y"],
          "phosphorylation_y": ["phosphorylation_st", "phosphorylation_y"],
          "sumoylation_k": ["sumoylation_k"],
          "ubiquitination_k": ["ubiquitination_k"]}
ORDER = ["phosphorylation_st", "phosphorylation_y", "acetylation_k",
         "methylation_k", "methylation_r", "sumoylation_k",
         "ubiquitination_k", "glycosylation_n"]
SHORT = {"phosphorylation_st": "Phospho S/T", "phosphorylation_y": "Phospho Y",
         "acetylation_k": "Acetylation K", "methylation_k": "Methylation K/R",
         "methylation_r": "Methylation R", "sumoylation_k": "Sumoylation K",
         "ubiquitination_k": "Ubiquitination K",
         "glycosylation_n": "N-Glyc N"}


def read_fasta(path):
    d = {}
    with open(path) as fh:
        for is_hdr, grp in groupby(fh, lambda l: l.startswith(">")):
            if is_hdr:
                acc = next(grp).split("|")[1]
            else:
                d[acc] = "".join(x.strip() for x in grp)
    return d


def positives():
    out = {}
    for t in TARGETS:
        df = pd.read_csv(f"{REBUILT}/{t}_all.tsv", sep="\t",
                         usecols=["protein", "pos", "y"])
        df = df[df.y == 1]
        d = defaultdict(set)
        for p, ps in zip(df.protein, df.pos):
            d[p].add(int(ps))
        out[t] = d
    return out


# measured c and the benchmark's own reading, from results_measured_c.txt and
# results_ptm_guarantee.txt; the scope under a measured c is task-specific and
# the worst-case scope is what remains without a reconstruction
MEASURED_C = {"phosphorylation_st": 0.0000, "phosphorylation_y": 0.0258,
              "acetylation_k": 0.0177, "methylation_k": 0.0474,
              "methylation_r": 0.1374, "sumoylation_k": 0.0033,
              "ubiquitination_k": 0.0134, "glycosylation_n": 0.0040}
DELTA_O = {"phosphorylation_st": 0.0165, "phosphorylation_y": 0.0532,
           "acetylation_k": 0.0578, "methylation_k": 0.0378,
           "methylation_r": 0.0534, "sumoylation_k": 0.0667,
           "ubiquitination_k": 0.0497, "glycosylation_n": 0.0100}


def main():
    fa = read_fasta(FASTA)
    pos = positives()

    print("Smallest annotation depth d_min such that, over entities with depth")
    print(">= d_min, the benchmark's unobserved weight falls to 0.5 or below.")
    print("Threshold T = %d.\n" % T)
    hdr = "%-20s %6s %8s %10s" % ("task", "d_min", "W", "% proteome")
    print("%s | %s | %s" % (hdr, hdr[21:], hdr[21:]))
    print("%-20s %26s | %26s | %26s"
          % ("", "(A) any gap, worst case", "(B) own gap, worst case",
             "(B) own gap, measured c"))
    print("-" * 108)

    for t in ORDER:
        res = set(TARGETS[t])
        depth = defaultdict(int)
        for m in MERGED[t]:
            for p, s in pos[m].items():
                depth[p] += len(s)

        dep, neg = [], []
        for p, seq in fa.items():
            c = sum(1 for ch in seq if ch in res)
            if c <= 0:
                continue
            k = len(pos[t].get(p, ()))
            dep.append(depth.get(p, 0))
            neg.append(max(c - k, 0))
        dep = np.array(dep)
        neg = np.array(neg, dtype=float)
        n_all, mass_all = len(dep), neg.sum()

        c = MEASURED_C.get(t)
        d_O = abs(DELTA_O.get(t, 0.0))
        # Both columns must ask the same question, and the question is whether
        # THIS task's own measured gap is robust.  The worst case admits
        # |delta_bar_U - delta_O| <= 1 + delta_O, so the condition is
        # W <= delta_O/(1 + delta_O); the measured one replaces 1 + delta_O by
        # c.  Using W <= 0.5 for the first column instead -- the condition for
        # *some* conceivable gap to be identified -- compares two different
        # questions and can make measurement look more restrictive than the
        # worst case, which it never is at a fixed gap.
        w_target_worst = d_O / (1.0 + d_O)
        w_target_meas = float("inf") if not c else d_O / c

        def first_below(limit):
            for d_min in range(0, T + 1):
                keep = dep >= d_min
                if not keep.any() or neg[keep].sum() == 0:
                    return None
                w = neg[keep & (dep < T)].sum() / neg[keep].sum()
                if w <= limit:
                    return d_min, w, 100 * keep.sum() / n_all
            return None

        a = first_below(0.5)
        b0 = first_below(w_target_worst)
        b = first_below(w_target_meas)
        def fmt(r):
            return ("%6s %8s %10s" % (">T", "--", "--")) if r is None                 else ("%6d %8.3f %9.1f%%" % r)
        print("%-20s %s | %s | %s"
              % (SHORT[t], fmt(a), fmt(b0), fmt(b)))
    print("-" * 78)
    print("\nd_min = T means the only population over which the guarantee is")
    print("available is the donor pool itself -- the benchmark supports")
    print("comparisons about the entities it sampled from and no others.")


if __name__ == "__main__":
    main()
