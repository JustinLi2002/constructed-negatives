#!/usr/bin/env python3
"""Reference-population coverage screens, not measured ranking certificates.

(A) finds the first depth with W <= 0.5. This only permits a sufficiently
large observed effect to pass an unrestricted sign bound. (B) illustrates
the stronger bound using a published gap, conditional on zero within-O shift.
Measured reconstruction effects must instead be recomputed within each split
and negative target by scope_reanalysis.py. No fixed c is used here.

PTM_AUDIT_BASE=/path/to/HRP python deployment_scope.py
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


def main():
    # Imported here to keep the shared annotation/FASTA definitions importable.
    from pathlib import Path
    from scope_reanalysis import TASKS, PUBLISHED, geometry, load_annotations
    _, positive, depths = load_annotations(Path(BASE))
    fasta = read_fasta(FASTA)
    print('Reference scope: A is a coverage screen; B assumes Delta_bench=Delta_O.')
    print('task\td_min_A\tW_reference\tA_pct\tB_scenario_pct')
    for task in TASKS:
        _, rows = geometry(fasta, positive[task], depths[task], task, T)
        a = next(r for r in rows if r['W'] <= .5)
        limit = PUBLISHED[task]/(1+PUBLISHED[task])
        b = next(r for r in rows if r['W'] <= limit)
        print(f"{task}\t{a['d_min']}\t{rows[0]['W']:.6f}\t"
              f"{a['retained_protein_pct']:.6f}\t{b['retained_protein_pct']:.6f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', action='store_true',
                        help='Legacy flag: reference screens only; measured effects are in scope_reanalysis.py')
    parser.parse_args()
    main()
