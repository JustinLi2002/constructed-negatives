#!/usr/bin/env python3
"""Does degree-distribution-balanced negative sampling violate positivity?

DDBSampling (Li, Shao, Zhao & Liu, BMC Biol 2025;23:123, code at
github.com/zpliulab/DDBSampling) is the corrective rule in the taxonomy: random
pairing under-represents hubs among the negatives, so a classifier learns that
high degree means interaction, and DDB is meant to remove exactly that.  It is
therefore the interesting case.  A rule designed to fix degree bias may still
leave some region of the degree axis with zero negatives, and if it does, it
fixes the bias a model can see while creating one it cannot.

Their algorithm, from `DDB_sampling` in prepare_dataset.py: index every node
pair by its **degree sum** d_u + d_v; then for each positive edge, take an
unused pair with the same degree sum, expanding outward to neighbouring sums
when that bucket is exhausted.  One negative per positive.

Two departures, both noted rather than silently fixed.  Their `cal_edge_degree`
enumerates all `nodes x nodes` pairs without excluding `mo1 == mo2`, and a
self-pair is not in the positive set, so a protein paired with itself can be
emitted as a negative; this excludes self-pairs.  And their exhaustive
enumeration is replaced by sampling a degree composition (a, s-a) with
probability proportional to h[a] * h[s-a] and then drawing nodes uniformly,
which is the same distribution over pairs without materialising n^2 of them.

Random sampling is reproduced alongside as the control, since the paper's whole
comparison is DDB against random.

    python ddb_sampling.py edges.tsv --out string700_ddb
"""
import argparse
import sys

import numpy as np
from scipy import sparse

from cl3_exclusion import DEG_EDGES, load_edges


def edge_set(A):
    """Undirected edges as a set of (lo, hi) index tuples."""
    U = sparse.triu(A, k=1).tocoo()
    return set(zip(U.row.tolist(), U.col.tolist())), U.row, U.col


def ddb_negatives(A, rng, max_tries=200):
    """One negative per positive, matched on degree sum with outward fallback."""
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    edges, er, ec = edge_set(A)

    by_deg = {}
    for d in np.unique(deg):
        by_deg[int(d)] = np.where(deg == d)[0]
    degs_present = np.array(sorted(by_deg))

    # for each achievable sum, the compositions (a, b) and their pair counts,
    # which is what makes a uniform draw over pairs of that sum cheap
    comps = {}
    for a in degs_present:
        for b in degs_present:
            if b < a:
                continue
            s = int(a + b)
            na, nb = len(by_deg[int(a)]), len(by_deg[int(b)])
            cnt = na * (na - 1) // 2 if a == b else na * nb
            if cnt:
                comps.setdefault(s, []).append((int(a), int(b), cnt))
    sums_present = np.array(sorted(comps))
    for s in comps:
        w = np.array([c[2] for c in comps[s]], dtype=float)
        comps[s] = (comps[s], w / w.sum())

    used = set()
    neg = []

    def draw(s):
        """A uniformly random node pair of degree sum s, or None."""
        if s not in comps:
            return None
        opts, p = comps[s]
        for _ in range(max_tries):
            a, b, _ = opts[rng.choice(len(opts), p=p)]
            u = by_deg[a][rng.integers(len(by_deg[a]))]
            v = by_deg[b][rng.integers(len(by_deg[b]))]
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in edges or key in used:
                continue
            return key
        return None

    order = np.argsort(sums_present)
    pos_of = {int(s): i for i, s in enumerate(sums_present[order])}

    for k in range(len(er)):
        s0 = int(deg[er[k]] + deg[ec[k]])
        got = draw(s0)
        if got is None:
            # expand outward through the achievable sums, nearest first
            i0 = pos_of.get(s0)
            if i0 is None:
                i0 = int(np.searchsorted(sums_present, s0))
            for off in range(1, len(sums_present)):
                for j in (i0 - off, i0 + off):
                    if 0 <= j < len(sums_present):
                        got = draw(int(sums_present[j]))
                        if got is not None:
                            break
                if got is not None:
                    break
        if got is None:
            continue
        used.add(got)
        neg.append(got)
        if len(neg) % 20000 == 0:
            print('  %d / %d negatives' % (len(neg), len(er)), end='\r',
                  file=sys.stderr, flush=True)
    print(' ' * 40, end='\r', file=sys.stderr)
    return np.array(neg), deg


def random_negatives(A, rng):
    """The control: uniformly random non-adjacent pairs, one per positive."""
    n = A.shape[0]
    edges, er, _ = edge_set(A)
    used, neg = set(), []
    while len(neg) < len(er):
        u, v = rng.integers(n), rng.integers(n)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in edges or key in used:
            continue
        used.add(key)
        neg.append(key)
    return np.array(neg)


def report(name, neg, deg, n):
    counts = np.zeros(n, dtype=np.int64)
    if len(neg):
        np.add.at(counts, neg[:, 0], 1)
        np.add.at(counts, neg[:, 1], 1)

    print('\n%s: %d negatives' % (name, len(neg)))
    print('  %10s %8s %10s %12s %12s'
          % ('degree', 'n', 'zero neg', 'mean neg', 'neg / pos'))
    print('  ' + '-' * 56)
    rows = []
    for i in range(len(DEG_EDGES) - 1):
        lo, hi = DEG_EDGES[i], DEG_EDGES[i + 1]
        m = (deg >= lo) & (deg < hi)
        if not m.any():
            continue
        lab = '%d-%d' % (lo, hi - 1) if np.isfinite(hi) else '%d+' % lo
        ratio = counts[m].sum() / max(deg[m].sum(), 1)
        rows.append((lab, int(m.sum()), float((counts[m] == 0).mean()),
                     float(counts[m].mean()), float(ratio)))
        print('  %10s %8d %9.1f%% %12.2f %12.3f'
              % (lab, m.sum(), 100 * (counts[m] == 0).mean(), counts[m].mean(),
                 ratio))
    print('  ' + '-' * 56)
    print('  proteins with zero negatives: %d of %d (%.1f%%)'
          % (int((counts == 0).sum()), n, 100 * (counts == 0).mean()))
    print('  of those, carrying %.3f of all degree'
          % (deg[counts == 0].sum() / deg.sum()))
    return counts, rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('edges')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='ddb')
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    A, names = load_edges(a.edges)
    n = A.shape[0]
    print('loaded %d proteins, %d edges' % (n, A.nnz // 2))

    neg_r = random_negatives(A, rng)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    c_r, rows_r = report('random sampling (control)', neg_r, deg, n)

    neg_d, deg = ddb_negatives(A, rng)
    c_d, rows_d = report('DDB sampling (degree-sum matched)', neg_d, deg, n)

    print('\nzero neg = share of proteins in that stratum that contribute '
          'positives\nbut no negatives at all.  A stratum at 100% is a '
          'positivity violation.\nneg / pos = negatives contributed per '
          'positive contributed; systematic\nvariation without a zero is '
          'covariate shift, which is repairable.')

    with open(a.out + '_by_degree.tsv', 'w', encoding='utf-8') as fh:
        fh.write('rule\tstratum\tn\tshare_zero_neg\tmean_neg\tneg_per_pos\n')
        for tag, rows in (('random', rows_r), ('ddb', rows_d)):
            for r in rows:
                fh.write('%s\t%s\t%d\t%.6f\t%.3f\t%.6f\n' % ((tag,) + r))
    with open(a.out + '_per_protein.tsv', 'w', encoding='utf-8') as fh:
        fh.write('protein\tdegree\tn_neg_random\tn_neg_ddb\n')
        for i in range(n):
            fh.write('%s\t%d\t%d\t%d\n' % (names[i], deg[i], c_r[i], c_d[i]))
    print('\nwritten: %s_by_degree.tsv, %s_per_protein.tsv' % (a.out, a.out))
