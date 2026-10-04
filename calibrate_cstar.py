#!/usr/bin/env python3
"""Put the break-even difference c* on an observable scale.

c* = |Delta_O|/W is the regional difference that would overturn the observed
sign. On its own it has no reference point. Following benchmark-based
sensitivity analysis, compare it with heterogeneity that is already visible
inside the sampled region: split each task-split's observed negatives (depth
>= T) into three depth tiers of roughly equal negative count, keeping each
protein in one tier, and compute each tier's effect Delta_k with positives
fixed. h = max_k |Delta_k - Delta_O| is the largest deviation an observed
stratum shows from the observed-region mean.

c*/h needs only the sampled region, so a benchmark can report it without a
reconstruction. The reconstruction then supplies c, which is reported against
the same scale. Tiers are a design choice; their depth ranges and sizes are
written out so the scale can be inspected.

python calibrate_cstar.py --base ~/HRP --out results_calibration
python calibrate_cstar.py --self-test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scope_reanalysis import (TASKS, SCORES, load_annotations, negative_contribution,
                              paired_predictions)


def depth_tiers(protein, depth, k=3):
    """Assign observed negatives to k tiers by protein depth, whole proteins only.

    Proteins are ordered by depth (then name, for determinism) and cut where the
    cumulative negative count crosses j/k of the total.
    """
    frame = pd.DataFrame({'protein': protein, 'depth': depth})
    per = frame.groupby('protein').agg(depth=('depth', 'first'), n=('depth', 'size'))
    per = per.reset_index().sort_values(['depth', 'protein'], kind='mergesort').reset_index(drop=True)
    cum = per.n.cumsum().to_numpy() / per.n.sum()
    per['tier'] = np.minimum((np.ceil(cum * k) - 1).astype(int), k - 1)
    # ceil-based cuts can leave a tier empty when one protein spans a boundary
    per['tier'] = pd.factorize(per.tier, sort=True)[0]
    return frame.protein.map(dict(zip(per.protein, per.tier))).to_numpy(), per


def calibrate(g, depth, protein, threshold, k=3):
    obs = depth >= threshold
    if not obs.any():
        return None
    do = float(g[obs].mean())
    du = float(g[~obs].mean()) if (~obs).any() else do
    W = float((~obs).mean())
    tier, per = depth_tiers(protein[obs], depth[obs], k)
    rows = []
    for t in np.unique(tier):
        sel = tier == t
        members = per[per.tier == t]
        rows.append(dict(tier=int(t), effect=float(g[obs][sel].mean()), n_neg=int(sel.sum()),
                         n_proteins=len(members), depth_min=int(members.depth.min()),
                         depth_max=int(members.depth.max())))
    h = max(abs(r['effect'] - do) for r in rows)
    c, cstar = abs(du - do), abs(do) / W
    return dict(W_R=W, delta_O=do, delta_U=du, c=c, c_star=cstar, h=h,
                n_tiers=len(rows), c_star_over_h=cstar / h if h > 0 else np.inf,
                c_over_h=c / h if h > 0 else np.inf, delta_R=float(g.mean())), rows


def main(args):
    base = Path(args.base).expanduser()
    root = Path(args.root).expanduser() if args.root else base / 'pdisjoint_runs_v2'
    _, _, depths = load_annotations(base)
    runs, tiers = [], []
    for task in TASKS:
        for split in range(args.splits):
            m, _ = paired_predictions(root, task, 'esm2', split, 'replica', 'rebuilt')
            if m is None:
                raise FileNotFoundError(f'Missing prediction pair for {task}/split{split}')
            dep = m.protein.map(depths[task]).fillna(0).to_numpy(int)
            neg = m.y.to_numpy() == 0
            pos = ~neg
            for score in SCORES:
                g = (negative_contribution(m.loc[pos, score + '_aug'], m.loc[neg, score + '_aug'])
                     - negative_contribution(m.loc[pos, score + '_base'], m.loc[neg, score + '_base']))
                for k in (2, 3, 4):   # 3 is reported; 2 and 4 check sensitivity to the tiering
                    out = calibrate(g, dep[neg], m.loc[neg, 'protein'].to_numpy(), args.threshold, k)
                    if out is None:
                        continue
                    vals, rows = out
                    runs.append(dict(task=task, split=split, score=score, k=k, **vals))
                    for r in rows:
                        tiers.append(dict(task=task, split=split, score=score, k=k, **r))
        print(task, 'complete', flush=True)
    rf, tf = pd.DataFrame(runs), pd.DataFrame(tiers)
    prefix = str(Path(args.out))
    rf.to_csv(prefix + '_runs.tsv', sep='\t', index=False, float_format='%.6g')
    tf.to_csv(prefix + '_tiers.tsv', sep='\t', index=False, float_format='%.6g')
    e = rf[(rf.score == 'y_pred') & (rf.k == 3)]
    summary = e.groupby('task', sort=False)[['W_R', 'delta_O', 'c', 'c_star', 'h',
                                             'c_star_over_h', 'c_over_h', 'delta_R']].mean()
    summary['h_min'] = e.groupby('task', sort=False).h.min()
    summary['h_max'] = e.groupby('task', sort=False).h.max()
    summary.reset_index().to_csv(prefix + '_summary.tsv', sep='\t', index=False, float_format='%.6g')
    print(summary.to_string(float_format=lambda x: f'{x:.4f}'))
    sens = rf[rf.score == 'y_pred'].groupby(['task', 'k'], sort=False).c_star_over_h.mean().unstack()
    print('c*/h by number of tiers:')
    print(sens.to_string(float_format=lambda x: f'{x:.3f}'))


def self_test():
    rng = np.random.default_rng(1)
    # Tiers keep proteins whole and are ordered by depth.
    prot = np.repeat(list('abcdef'), [5, 1, 4, 2, 6, 3])
    dep = np.repeat([30, 10, 50, 12, 90, 20], [5, 1, 4, 2, 6, 3])
    tier, per = depth_tiers(prot, dep)
    assert all(len(set(tier[prot == p])) == 1 for p in set(prot))
    order = per.sort_values('depth')
    assert list(order.tier) == sorted(order.tier)
    # h is zero when every observed stratum has the same effect.
    g = np.r_[np.full(21, .05), rng.normal(-.1, .01, 30)]
    d = np.r_[dep, np.zeros(30, int)]
    p = np.r_[prot, np.array(['u%d' % i for i in range(30)])]
    vals, _ = calibrate(g, d, p, 10)
    assert vals['h'] < 1e-12 and abs(vals['c_star'] - .05 / (30 / 51)) < 1e-12
    # Unit-weight effects agree with the scope_reanalysis identity.
    assert abs(vals['delta_R'] - ((1 - vals['W_R']) * vals['delta_O'] + vals['W_R'] * vals['delta_U'])) < 1e-12
    print('calibrate_cstar self-tests: 3 passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='~/HRP')
    ap.add_argument('--root')
    ap.add_argument('--out', default='results_calibration')
    ap.add_argument('--splits', type=int, default=3)
    ap.add_argument('--threshold', type=int, default=10)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    self_test() if a.self_test else main(a)
