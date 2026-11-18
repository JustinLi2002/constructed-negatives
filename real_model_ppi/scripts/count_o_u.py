"""Label-free counts for the planned real-model PPI test on STRING >=700.
No model scores are read.  Uses the manuscript's own prefilter helpers."""
import sys
import numpy as np
sys.path.insert(0, r"<REPO_ROOT>")
from cl3_exclusion import load_edges, config_threshold

A, names = load_edges(r"<REPO_ROOT>/data/string_phys_700.tsv")
deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
n, m = len(deg), A.nnz // 2
tau = config_threshold(deg, 0.064)

# unordered pairs u<v with d_u*d_v <= tau, via the degree histogram
vals, cnts = np.unique(deg, return_counts=True)
pairs_le = 0
for i, (da, ca) in enumerate(zip(vals, cnts)):
    for db, cb in zip(vals[i:], cnts[i:]):
        if da * db > tau:
            break
        pairs_le += ca * (ca - 1) // 2 if da == db else ca * cb
U_, V_ = A.nonzero()
mask = U_ < V_
edge_prod = deg[U_[mask]] * deg[V_[mask]]
edges_le = int((edge_prod <= tau).sum())
eligible = pairs_le - edges_le
nonedges = n * (n - 1) // 2 - m
W = 1 - eligible / nonedges

# proteins with at least one eligible (non-adjacent, product <= tau) partner
low = np.flatnonzero(deg <= tau)
has_partner = 0
for u in low:
    cand = np.flatnonzero(deg * deg[u] <= tau)
    cand = cand[cand != u]
    nb = set(A.indices[A.indptr[u]:A.indptr[u + 1]].tolist())
    if any(v not in nb for v in cand[:len(nb) + 1]):
        has_partner += 1
deg_hist = {int(d): int((deg == d).sum()) for d in range(1, 6)}
print(f"proteins {n}, edges {m}, tau {tau}")
print(f"degree histogram 1..5: {deg_hist}")
print(f"non-edge pairs {nonedges:,}; eligible non-edges (O) {eligible:,}; excluded (U) {nonedges - eligible:,}")
print(f"W (full non-edge target, uniform) = {W:.4f}")
print(f"proteins with >=1 eligible partner: {has_partner} of {n}")
print(f"positive edges with product <= tau: {edges_le}; median edge degree product {int(np.median(edge_prod))}")
