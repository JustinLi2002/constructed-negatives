#!/usr/bin/env python3
"""Scope statistics for k-core-filtered recommendation benchmarks (Amazon Reviews 2023).

The five-core rule is the recommender counterpart of the PTM donor threshold: a
user-item pair is eligible only if both its user and its item lie in the 5-core of
the bipartite interaction graph.

Target   : unobserved (user, item) pairs among all users and items of a category
           (the 0-core release), weighted uniformly
Depth    : of a pair = min(core number of its user, core number of its item)
Rule     : the 5-core keeps pairs with depth >= T = 5
W        : share of the target's candidate negatives with depth < T
W_d      : the same share for the depth->=d target, mass[d <= depth < T] / mass[depth >= d]
d_min    : smallest d with W_d <= 0.5 (the family of depth-restricted targets, as for PTM)

Inputs   : <data_dir>/0core/<Category>.csv.gz and <data_dir>/5core/<Category>.csv.gz,
           the rating_only benchmark files from
           https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/{0core,5core}/rating_only/
Validation: the 5-core recomputed from the 0-core file must equal the official 5-core file
           (same users, items and user-item pairs); the script stops otherwise.
Usage    : python recsys_kcore_scope.py <data_dir> <out.tsv> <Category> [<Category> ...]
"""
import csv, gzip, sys
from pathlib import Path

import numpy as np

T = 5


def load_pairs(path):
    """Distinct user-item pairs (duplicate ratings collapsed) and the raw row count."""
    users, items, uu, ii = {}, {}, [], []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        assert next(r)[:2] == ["user_id", "parent_asin"]
        for row in r:
            uu.append(users.setdefault(row[0], len(users)))
            ii.append(items.setdefault(row[1], len(items)))
    e = np.unique(np.array(uu, dtype=np.int64) * len(items) + np.array(ii, dtype=np.int64))
    return users, items, e // len(items), e % len(items), len(uu)


def core_numbers(nu, ni, eu, ei):
    """Bipartite k-core number of every user and item, by nested iterative pruning."""
    cu, ci = np.zeros(nu, np.int32), np.zeros(ni, np.int32)
    alive_u, alive_i = np.ones(nu, bool), np.ones(ni, bool)
    k = 1
    while True:
        while True:
            m = alive_u[eu] & alive_i[ei]
            du, di = np.bincount(eu[m], minlength=nu), np.bincount(ei[m], minlength=ni)
            drop_u, drop_i = alive_u & (du < k), alive_i & (di < k)
            if not (drop_u.any() or drop_i.any()):
                break
            alive_u &= ~drop_u; alive_i &= ~drop_i
        if not (alive_u.any() and alive_i.any()):
            return cu, ci
        cu[alive_u] = k; ci[alive_i] = k
        k += 1


def mass(d, cu, ci, eu, ei):
    """Unobserved pairs with depth >= d, and the users, items and observed pairs at that depth."""
    U, I = int((cu >= d).sum()), int((ci >= d).sum())
    E = int(((cu[eu] >= d) & (ci[ei] >= d)).sum())
    return U * I - E, U, I, E


def main(data_dir, out, cats):
    D = Path(data_dir)
    cols = ["category", "ratings", "pairs", "users", "items", "core_users", "core_items", "core_pairs",
            "official_match", "W", "W_d4", "d_min", "retained_users_pct", "retained_items_pct",
            "retained_mass_pct", "max_core"]
    rows = []
    for c in cats:
        users, items, eu, ei, nraw = load_pairs(D / "0core" / f"{c}.csv.gz")
        cu, ci = core_numbers(len(users), len(items), eu, ei)
        ou, oi, oeu, _, _ = load_pairs(D / "5core" / f"{c}.csv.gz")
        inv_u, inv_i = list(users), list(items)          # dict order = integer id
        ours_u = {inv_u[x] for x in np.flatnonzero(cu >= T)}
        ours_i = {inv_i[x] for x in np.flatnonzero(ci >= T)}
        M = {d: mass(d, cu, ci, eu, ei) for d in range(1, T + 1)}
        match = ours_u == set(ou) and ours_i == set(oi) and M[T][3] == len(oeu)
        assert match, f"{c}: recomputed 5-core differs from the official release"
        obs = M[T][0]
        W_d = {d: (M[d][0] - obs) / M[d][0] for d in range(1, T + 1)}
        d_min = min(d for d in W_d if W_d[d] <= 0.5)
        rows.append([c, nraw, len(eu), len(users), len(items), M[T][1], M[T][2], M[T][3], match,
                     f"{W_d[1]:.6f}", f"{W_d[4]:.4f}", d_min, f"{100 * M[d_min][1] / len(users):.2f}",
                     f"{100 * M[d_min][2] / len(items):.2f}", f"{100 * M[d_min][0] / M[1][0]:.2f}", int(cu.max())])
        print("\t".join(map(str, rows[-1])), flush=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols); w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
