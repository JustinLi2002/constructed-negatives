#!/usr/bin/env python3
"""What the donor threshold buys and what it costs, on the same axis.

The threshold is not arbitrary.  A protein with few recorded sites may simply
not have been assayed, so its unmodified residues are unreliable negatives;
restricting negatives to well-annotated donors is a false-negative control, and
that is a legitimate motive the rest of this work has not engaged with.

Both sides of that trade are computable from the same two inputs the design
statistic needs -- a candidate-residue count per protein and its annotation
depth -- so they can be put on one axis.

  cost    W_un(T), the share of deployment negatives lying on proteins below
          the threshold, about which the benchmark carries no information.

  benefit FN(T), the expected share of the benchmark's negatives that are in
          fact unrecorded positives.  A site observed as negative at depth d is
          truly positive with posterior q(d); FN(T) is the candidate-weighted
          mean of q over the donor pool at T.  q comes from the first paper's
          fn_sensitivity.py, reused verbatim: with f(d) the observed positive
          rate among candidates at depth d and f_ref its maximum across strata,

              g  = f_ref * (f/f_ref) ** alpha
              pi = (f/f_ref) ** (1 - alpha)
              q  = g(1 - pi) / [g(1 - pi) + (1 - g)]

          alpha = 1 means no false negatives at all; alpha = 0 attributes the
          entire depth-rate relationship to missing annotation, which is the
          adversarial extreme and the most favourable case the threshold can be
          given.  Anchoring on the maximum rather than the deepest stratum is
          the first paper's choice and is deliberate: the rate is not monotone
          in depth, and every other anchor implies a smaller false-negative
          burden.

If the benefit curve flattens while the cost curve climbs, the threshold is
buying less than it destroys, and the crossing point says where.  That is the
only prescriptive statement this work can make to someone building a benchmark.

Run where the reconstructions live:

    PTM_AUDIT_BASE=/home/FCAM/juli/HRP python threshold_tradeoff.py
"""
import os
from collections import defaultdict
from itertools import groupby

import numpy as np

DIAG = bool(os.environ.get('DIAG'))
import pandas as pd

BASE = os.environ.get("PTM_AUDIT_BASE", "/home/FCAM/juli/HRP")
FASTA = f"{BASE}/deepmvp/DeepMVP/data/swiss_prot_human_20190214.fasta"
REBUILT = f"{BASE}/rebuilt"

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

DEPTH_EDGES = [0, 1, 2, 3, 5, 10, 20, 50, np.inf]
T_GRID = [0, 1, 2, 3, 5, 10, 20, 50]


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


def posterior(f, alpha):
    """q per depth bin.  Verbatim from the first paper's fn_sensitivity.py."""
    f = np.asarray(f, dtype=float)
    ref = int(np.argmax(f))
    f_ref = f[ref]
    if f_ref <= 0:
        return np.zeros_like(f), ref
    ratio = f / f_ref
    with np.errstate(divide="ignore", invalid="ignore"):
        g = f_ref * ratio ** alpha
        pi = ratio ** (1.0 - alpha)
    g = np.nan_to_num(g, nan=f_ref)
    pi = np.nan_to_num(pi, nan=0.0)
    q = g * (1 - pi) / (g * (1 - pi) + (1 - g))
    return q, ref


def curves(task, fa, pos, alpha):
    res = set(TARGETS[task])
    depth = defaultdict(int)
    for m in MERGED[task]:
        for p, s in pos[m].items():
            depth[p] += len(s)

    prot, dep, cand, npos = [], [], [], []
    for p, seq in fa.items():
        c = sum(1 for ch in seq if ch in res)
        if c <= 0:
            continue
        k = len(pos[task].get(p, ()))
        prot.append(p)
        dep.append(depth.get(p, 0))
        cand.append(c)
        npos.append(k)
    dep = np.array(dep)
    cand = np.array(cand, dtype=float)
    npos = np.array(npos, dtype=float)
    neg = np.maximum(cand - npos, 0.0)          # deployment negatives per protein

    b = np.digitize(dep, DEPTH_EDGES[1:-1], right=False)
    nb = len(DEPTH_EDGES) - 1
    f = np.array([npos[b == k].sum() / cand[b == k].sum() if (b == k).any()
                  and cand[b == k].sum() > 0 else 0.0 for k in range(nb)])
    q, ref = posterior(f, alpha)

    rows = []
    for T in T_GRID:
        donor = dep >= T
        w_un = neg[~donor].sum() / neg.sum() if neg.sum() else np.nan
        dn = neg[donor]
        fn = (dn * q[b[donor]]).sum() / dn.sum() if dn.sum() else np.nan
        rows.append((T, w_un, fn))
    return rows, f, q, ref, b, neg, dep


def main():
    fa = read_fasta(FASTA)
    pos = positives()
    for alpha in (0.0, 0.5):
        print("=" * 78)
        print("alpha = %.1f  %s" % (alpha, "(adversarial extreme: the whole "
              "depth-rate relationship is missing annotation)" if alpha == 0
              else "(half of it is)"))
        print("=" * 78)
        print("%-20s %5s %9s %9s %11s" % ("task", "T", "W_un", "FN rate",
                                          "FN saved"))
        print("-" * 60)
        agg = defaultdict(list)
        for t in ORDER:
            rows, f, q, ref, b, neg, dep = curves(t, fa, pos, alpha)
            if DIAG:
                nb = len(DEPTH_EDGES) - 1
                print('   %-9s %10s %9s %9s %9s'
                      % ('depth bin', 'neg weight', 'f', 'q', 'n prot'))
                for k in range(nb):
                    m = b == k
                    if not m.any():
                        continue
                    lab = ('%d-%d' % (DEPTH_EDGES[k], DEPTH_EDGES[k+1]-1)
                           if np.isfinite(DEPTH_EDGES[k+1])
                           else '%d+' % DEPTH_EDGES[k])
                    print('   %-9s %10.4f %9.5f %9.5f %9d%s'
                          % (lab, neg[m].sum()/neg.sum(), f[k], q[k],
                             m.sum(), '  <- f_ref' if k == ref else ''))
                print()
            fn0 = rows[0][2]
            for T, w, fn in rows:
                agg[T].append((w, fn, fn0 - fn))
            for i, (T, w, fn) in enumerate(rows):
                print("%-20s %5d %9.3f %9.4f %11.4f"
                      % (SHORT[t] if i == 0 else "", T, w, fn, fn0 - fn))
            print()
        print("-" * 60)
        print("%-20s %5s %9s %9s %11s" % ("MEAN over tasks", "T", "W_un",
                                          "FN rate", "FN saved"))
        for T in T_GRID:
            w = float(np.mean([a[0] for a in agg[T]]))
            fn = float(np.mean([a[1] for a in agg[T]]))
            sv = float(np.mean([a[2] for a in agg[T]]))
            print("%-20s %5d %9.3f %9.4f %11.4f" % ("", T, w, fn, sv))
        print()


if __name__ == "__main__":
    main()
