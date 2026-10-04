#!/usr/bin/env python3
"""Protein-cluster bootstrap intervals for the reconstruction effects in Table 1(C).

Each task-split is resampled by protein: every site of a drawn protein, positive
or negative, enters with that protein's draw count, so within-protein dependence
is kept. Positives are resampled with the negatives because AUROC contributions
depend on them. The three protein-disjoint splits are bootstrapped independently
and averaged replicate by replicate, which gives an interval for the split mean
reported in Table 1. Intervals describe resampling variability of these test
proteins; they are not a full-proteome guarantee.

python bootstrap_effects.py --base ~/HRP --out results_bootstrap
python bootstrap_effects.py --self-test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scope_reanalysis import (TASKS, load_annotations, negative_contribution,
                              paired_predictions)

STATS = ['W_R', 'delta_O', 'delta_U', 'c', 'c_star', 'delta_R']


def weighted_contribution(positive_scores, negative_scores, wpos):
    """AUROC contribution of each negative when positives carry weights.

    wpos has shape (reps, n_positive); returns shape (reps, n_negative).
    Ties earn half credit, exactly as in negative_contribution.
    """
    order = np.argsort(positive_scores, kind='mergesort')
    p = np.asarray(positive_scores, float)[order]
    w = wpos[:, order]
    cw = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(w, axis=1)], axis=1)
    left = np.searchsorted(p, negative_scores, side='left')
    right = np.searchsorted(p, negative_scores, side='right')
    total = cw[:, -1:]
    return (total - cw[:, right] + .5 * (cw[:, right] - cw[:, left])) / total


def replicate_stats(g, wneg, observed):
    """Weighted versions of the quantities in scope_reanalysis.effect_at_scope."""
    wo, wu = wneg * observed, wneg * ~observed
    so, su = wo.sum(1), wu.sum(1)
    do = (wo * g).sum(1) / so
    du = np.where(su > 0, (wu * g).sum(1) / np.where(su > 0, su, 1), do)
    W = su / (so + su)
    dr = (wneg * g).sum(1) / (so + su)
    c = np.abs(du - do)
    return dict(W_R=W, delta_O=do, delta_U=du, c=c,
                c_star=np.abs(do) / np.where(W > 0, W, np.nan), delta_R=dr)


def bootstrap_pair(m, depth, threshold, reps, rng, chunk=100, score='y_pred'):
    neg = m.y.to_numpy() == 0
    pos = ~neg
    proteins, index = np.unique(m.protein.to_numpy(), return_inverse=True)
    observed = depth[neg] >= threshold
    ps_b, ps_a = m.loc[pos, score + '_base'].to_numpy(float), m.loc[pos, score + '_aug'].to_numpy(float)
    ns_b, ns_a = m.loc[neg, score + '_base'].to_numpy(float), m.loc[neg, score + '_aug'].to_numpy(float)
    out = {k: [] for k in STATS}
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        counts = rng.multinomial(len(proteins), np.full(len(proteins), 1 / len(proteins)), size=r)
        weight = counts[:, index].astype(float)
        wpos, wneg = weight[:, pos], weight[:, neg]
        g = (weighted_contribution(ps_a, ns_a, wpos) - weighted_contribution(ps_b, ns_b, wpos))
        for k, v in replicate_stats(g, wneg, observed).items():
            out[k].append(v)
        done += r
    return {k: np.concatenate(v) for k, v in out.items()}


def main(args):
    base = Path(args.base).expanduser()
    root = Path(args.root).expanduser() if args.root else base / 'pdisjoint_runs_v2'
    _, _, depths = load_annotations(base)
    rng = np.random.default_rng(args.seed)
    per_split, per_task = [], []
    for task in TASKS:
        draws = []
        for split in range(args.splits):
            m, _ = paired_predictions(root, task, 'esm2', split, 'replica', 'rebuilt')
            if m is None:
                raise FileNotFoundError(f'Missing prediction pair for {task}/split{split}')
            dep = m.protein.map(depths[task]).fillna(0).to_numpy(int)
            d = bootstrap_pair(m, dep, args.threshold, args.reps, rng)
            draws.append(d)
            neg = m.y.to_numpy() == 0
            row = dict(task=task, split=split, n_proteins=int(m.protein.nunique()),
                       n_O_proteins=int(m.loc[neg & (dep >= args.threshold), 'protein'].nunique()),
                       # A replicate that draws no observed-region protein has no
                       # Delta_O; it is left out of every percentile (nanpercentile).
                       n_reps_without_O=int(np.isnan(d['delta_O']).sum()),
                       reversal_fraction=float(np.mean((np.sign(d['delta_R']) != np.sign(d['delta_O']))[~np.isnan(d['delta_O'])])))
            for k in STATS:
                lo, hi = np.nanpercentile(d[k], [2.5, 97.5])
                row[k + '_lo'], row[k + '_hi'] = float(lo), float(hi)
            per_split.append(row)
        mean = {k: np.mean([d[k] for d in draws], axis=0) for k in STATS}
        row = dict(task=task, reps=args.reps)
        for k in STATS:
            lo, hi = np.nanpercentile(mean[k], [2.5, 97.5])
            row[k + '_lo'], row[k + '_hi'] = float(lo), float(hi)
        row['c_exceeds_c_star_fraction'] = float(np.mean(mean['c'] > mean['c_star']))
        per_task.append(row)
        print(task, 'complete', flush=True)
    prefix = str(Path(args.out))
    pd.DataFrame(per_split).to_csv(prefix + '_splits.tsv', sep='\t', index=False, float_format='%.6g')
    pd.DataFrame(per_task).to_csv(prefix + '_means.tsv', sep='\t', index=False, float_format='%.6g')
    print(pd.DataFrame(per_task)[['task', 'delta_O_lo', 'delta_O_hi', 'delta_R_lo', 'delta_R_hi',
                                   'c_lo', 'c_hi', 'c_star_lo', 'c_star_hi']].to_string(index=False))


def self_test():
    rng = np.random.default_rng(0)
    p, n = rng.normal(1, 1, 40), rng.normal(0, 1, 60)
    p[:5] = n[:5]  # ties
    unit = weighted_contribution(p, n, np.ones((1, len(p))))[0]
    np.testing.assert_allclose(unit, negative_contribution(p, n))
    # An integer weight equals duplicating that positive.
    w = np.ones(len(p)); w[3] = 3
    dup = np.concatenate([p, [p[3], p[3]]])
    np.testing.assert_allclose(weighted_contribution(p, n, w[None, :])[0], negative_contribution(dup, n))
    # Unit weights reproduce the unweighted identity and c* = |delta_O|/W.
    g = rng.normal(0, .1, 60); obs = rng.random(60) < .4
    s = replicate_stats(g[None, :], np.ones((1, 60)), obs)
    W = 1 - obs.mean(); do = g[obs].mean(); du = g[~obs].mean()
    np.testing.assert_allclose([s['W_R'][0], s['delta_O'][0], s['delta_U'][0], s['delta_R'][0], s['c_star'][0]],
                               [W, do, du, (1 - W) * do + W * du, abs(do) / W])
    print('bootstrap_effects self-tests: 3 passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='~/HRP')
    ap.add_argument('--root')
    ap.add_argument('--out', default='results_bootstrap')
    ap.add_argument('--splits', type=int, default=3)
    ap.add_argument('--threshold', type=int, default=10)
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261004)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    self_test() if a.self_test else main(a)
