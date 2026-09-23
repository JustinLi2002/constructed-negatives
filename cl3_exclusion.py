#!/usr/bin/env python3
"""Which proteins can contribute a negative under the Contrastive-L3 rule.

UPNA-PPI (Chatterjee et al., Bioinformatics 2025;41:btaf148) builds negatives by
scoring all human protein pairs with a configuration model on the degree
sequence, keeping the bottom 10M as candidates, and then filtering to pairs that
induce **zero length-3 paths** in the PPI network.  The paper states that this
"can only identify negative interactions for younger proteins", younger meaning
peripheral in a hyperbolic embedding -- but it reports no degree-stratified
analysis of its own negative set and makes no claim that any protein contributes
exactly zero.  That claim is ours to establish, which is what this does.

The test is cheaper than it looks.  The number of length-3 walks from u to v is
A^3[u][v].  Walks are not paths in general -- u->v->b->v and u->a->u->v are
walks with repeated nodes -- but every degenerate case requires the edge (u,v),
and with no self-loops there are no others.  Negative candidates are by
construction **non-interacting** pairs, so on exactly the pairs we care about

    number of length-3 paths from u to v  ==  A^3[u][v] .

So the rule reduces to a boolean question: v is eligible as a negative partner
for u iff u and v are non-adjacent, distinct, and unreachable by any walk of
length exactly three.  No 156M-row pair list is ever materialised; it is one
sparse boolean matrix cubed, blocked by rows to keep the dense intermediate
small.

Outputs describe marginal protein coverage: degree, eligible-partner count,
and complete protein exclusion. The legacy column named "propensity" is the
mean eligible-partner share with denominator n-1, not a pair's inclusion
probability. Positive marginal coverage does not establish pair-level support;
support_unit_audit.py reports excluded nonedge pairs separately.

    python cl3_exclusion.py edges.tsv --out huri
    python cl3_exclusion.py edges.tsv --col-a 0 --col-b 1 --header

Input is an undirected edge list: any delimited text file with two identifier
columns.  Self-loops and duplicate edges are dropped, and the graph is
symmetrised.
"""
import argparse
import sys

import numpy as np
from scipy import sparse

# Degree strata, chosen so that each is wide enough to hold a stable propensity
# estimate and the top one isolates the hubs the rule is suspected to exclude.
DEG_EDGES = [1, 2, 3, 5, 10, 25, 50, 100, 250, np.inf]


def load_edges(path, col_a=0, col_b=1, sep=None, header=False):
    """Read an undirected edge list into a symmetric CSR adjacency matrix."""
    pairs = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for i, line in enumerate(fh):
            if header and i == 0:
                continue
            line = line.rstrip('\n').rstrip('\r')
            if not line or line.startswith('#'):
                continue
            f = line.split(sep) if sep else line.split()
            if len(f) <= max(col_a, col_b):
                continue
            a, b = f[col_a].strip(), f[col_b].strip()
            if a and b and a != b:
                pairs.append((a, b))

    if not pairs:
        sys.exit('no edges parsed -- check --col-a/--col-b and --sep')

    names = sorted({x for p in pairs for x in p})
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    r = np.fromiter((idx[a] for a, _ in pairs), dtype=np.int32, count=len(pairs))
    c = np.fromiter((idx[b] for _, b in pairs), dtype=np.int32, count=len(pairs))
    A = sparse.csr_matrix((np.ones(len(pairs), np.int8), (r, c)), shape=(n, n))
    A = A + A.T                      # symmetrise
    A.setdiag(0)                     # no self-loops
    A.eliminate_zeros()
    A.data[:] = 1                    # collapse duplicate edges
    return A.astype(np.int8).tocsr(), names


def config_threshold(deg, frac):
    """Degree-product cutoff reproducing UPNA-PPI's configuration-model prefilter.

    The configuration model assigns an interaction probability that is monotone
    increasing in the degree product d_u * d_v, so taking the bottom-N least
    probable pairs is exactly taking the N pairs of smallest degree product.
    UPNA-PPI keeps the bottom 10M of ~156M human pairs; `frac` is that share,
    applied to whatever network is at hand.

    This is where the rule bites.  A protein of degree d can only reach the
    cutoff by pairing with partners of degree <= tau / d, so as d grows its set
    of admissible partners empties -- and for a hub it empties completely, which
    is a propensity of exactly zero rather than a small one.
    """
    n = len(deg)
    target = frac * n * (n - 1) / 2.0
    order = np.sort(deg)
    lo, hi = 1, int(deg.max()) ** 2
    while lo < hi:
        tau = (lo + hi) // 2
        # pairs with d_u * d_v <= tau, counted over ordered pairs then halved
        cnt = np.searchsorted(order, tau // np.maximum(deg, 1), side='right').sum()
        cnt = (cnt - np.count_nonzero(deg * deg <= tau)) / 2.0
        if cnt < target:
            lo = tau + 1
        else:
            hi = tau
    return lo


def eligible_counts(A, block=512, verbose=True, prefilter_frac=None):
    """For each protein, how many partners induce zero length-3 paths.

    Returns (eligible, n_pairs_possible) where eligible[u] counts the v that are
    distinct from u, non-adjacent to u, and have A^3[u][v] == 0.  With
    prefilter_frac set, v must additionally survive the configuration-model
    prefilter, which is the full UPNA-PPI rule rather than CL3 alone.
    """
    n = A.shape[0]
    Ab = A.astype(bool)
    eligible = np.zeros(n, dtype=np.int64)

    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    tau = None
    if prefilter_frac:
        tau = config_threshold(deg, prefilter_frac)
        print('  configuration-model cutoff: degree product <= %d' % tau,
              file=sys.stderr)

    for s in range(0, n, block):
        e = min(s + block, n)
        # length-2 then length-3 walks out of this row block; both stay sparse
        # because the product of the two is what densifies, not the operands
        w3 = (Ab[s:e] @ Ab) @ Ab
        reach = np.asarray(w3.todense()) if sparse.issparse(w3) else w3
        reach = reach > 0

        adj = np.asarray(Ab[s:e].todense())
        blocked = reach | adj
        blocked[np.arange(e - s), np.arange(s, e)] = True   # self
        if tau is not None:
            blocked |= (deg[s:e, None] * deg[None, :]) > tau
        eligible[s:e] = (~blocked).sum(axis=1)

        if verbose:
            print('  rows %6d / %d' % (e, n), end='\r', flush=True, file=sys.stderr)
    if verbose:
        print(' ' * 40, end='\r', file=sys.stderr)

    return eligible, n - 1


def summarise(A, names, eligible, out_prefix):
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)

    n_pairs_total = n * (n - 1) // 2
    n_edges = int(A.nnz // 2)
    # each eligible ordered pair is counted twice across the two endpoints
    n_eligible_pairs = int(eligible.sum() // 2)
    excluded = eligible == 0

    print('\nnetwork')
    print('  proteins                     %10d' % n)
    print('  edges                        %10d' % n_edges)
    print('  all pairs                    %10d' % n_pairs_total)
    print('  mean degree                  %10.1f' % deg.mean())

    print('\nunder the Contrastive-L3 rule')
    print('  pairs with zero L3 paths     %10d  (%.2f%% of all pairs)'
          % (n_eligible_pairs, 100 * n_eligible_pairs / n_pairs_total))
    print('  proteins that may appear     %10d' % int((~excluded).sum()))
    print('  proteins EXCLUDED outright   %10d  (%.1f%%)'
          % (int(excluded.sum()), 100 * excluded.mean()))

    # positive-edge degree mass: distinct from candidate-negative pair mass
    share_of_pair_mass = deg.astype(float) / deg.sum()
    print('  share of degree carried by excluded proteins  %.3f'
          % share_of_pair_mass[excluded].sum())

    print('\npropensity of appearing as a negative, by degree')
    print('  %10s %8s %10s %12s %10s'
          % ('degree', 'n', 'excluded', 'mean elig.', 'propensity'))
    print('  ' + '-' * 54)
    rows = []
    for i in range(len(DEG_EDGES) - 1):
        lo, hi = DEG_EDGES[i], DEG_EDGES[i + 1]
        m = (deg >= lo) & (deg < hi)
        if not m.any():
            continue
        lab = '%d-%d' % (lo, hi - 1) if np.isfinite(hi) else '%d+' % lo
        prop = eligible[m].mean() / (n - 1)
        rows.append((lab, int(m.sum()), float(excluded[m].mean()),
                     float(eligible[m].mean()), float(prop)))
        print('  %10s %8d %9.1f%% %12.0f %10.4f'
              % (lab, m.sum(), 100 * excluded[m].mean(), eligible[m].mean(),
                 prop))
    print('  ' + '-' * 54)
    print('\npropensity = mean share of all possible partners a protein of that'
          '\ndegree may legally take as a negative.  A stratum at exactly 0 is a'
          '\nprotein-level exclusion; positive marginal eligibility does not establish'
          '\nfull pair support. Check the evaluated unit separately.')

    with open(out_prefix + '_per_protein.tsv', 'w', encoding='utf-8') as fh:
        fh.write('protein\tdegree\tn_eligible\texcluded\n')
        for i in range(n):
            fh.write('%s\t%d\t%d\t%d\n'
                     % (names[i], deg[i], eligible[i], int(excluded[i])))
    with open(out_prefix + '_by_degree.tsv', 'w', encoding='utf-8') as fh:
        fh.write('stratum\tn\tshare_excluded\tmean_eligible\tpropensity\n')
        for r in rows:
            fh.write('%s\t%d\t%.6f\t%.3f\t%.6f\n' % r)
    print('\nwritten: %s_per_protein.tsv, %s_by_degree.tsv'
          % (out_prefix, out_prefix))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('edges')
    ap.add_argument('--col-a', type=int, default=0)
    ap.add_argument('--col-b', type=int, default=1)
    ap.add_argument('--sep', default=None)
    ap.add_argument('--header', action='store_true')
    ap.add_argument('--block', type=int, default=512)
    ap.add_argument('--out', default='cl3')
    ap.add_argument('--prefilter-frac', type=float, default=None,
                    help='share of all pairs kept by the configuration-model '
                         'prefilter; UPNA-PPI uses 10M/156M = 0.064')
    a = ap.parse_args()

    A, names = load_edges(a.edges, a.col_a, a.col_b, a.sep, a.header)
    print('loaded %d proteins, %d edges' % (A.shape[0], A.nnz // 2))
    eligible, _ = eligible_counts(A, block=a.block,
                                  prefilter_frac=a.prefilter_frac)
    summarise(A, names, eligible, a.out)
