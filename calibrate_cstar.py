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
    # SD-based alternative: negative-weighted RMS deviation of tier effects from
    # Delta_O; less driven by a single extreme tier than the maximum.
    sd = float(np.sqrt(sum(r['n_neg'] * (r['effect'] - do) ** 2 for r in rows)
                       / sum(r['n_neg'] for r in rows)))
    c, cstar = abs(du - do), abs(do) / W
    return dict(W_R=W, delta_O=do, delta_U=du, c=c, c_star=cstar, h=h, sd=sd,
                n_tiers=len(rows), c_star_over_h=cstar / h if h > 0 else np.inf,
                c_over_h=c / h if h > 0 else np.inf,
                c_star_over_sd=cstar / sd if sd > 0 else np.inf,
                c_over_sd=c / sd if sd > 0 else np.inf, delta_R=float(g.mean())), rows


def bootstrap_scale(m, depth, threshold, reps, rng, ks=(2, 3, 4), chunk=100, score='y_pred'):
    """Protein bootstrap of h, sd, c and c* with tiers held at their original assignment.

    The draws reproduce bootstrap_effects.bootstrap_pair exactly when called with
    the same generator state, so replicates pair with Extended Data Table 2.
    """
    from bootstrap_effects import weighted_contribution
    neg = m.y.to_numpy() == 0
    pos = ~neg
    proteins, index = np.unique(m.protein.to_numpy(), return_inverse=True)
    obs = depth[neg] >= threshold
    pobs = m.loc[neg, 'protein'].to_numpy()[obs]
    tiers = {k: depth_tiers(pobs, depth[neg][obs], k)[0] for k in ks}
    ps_b, ps_a = m.loc[pos, score + '_base'].to_numpy(float), m.loc[pos, score + '_aug'].to_numpy(float)
    ns_b, ns_a = m.loc[neg, score + '_base'].to_numpy(float), m.loc[neg, score + '_aug'].to_numpy(float)
    out = {key: [] for key in ['c', 'c_star'] + [f'{v}_{k}' for k in ks for v in ('h', 'sd')]}
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        counts = rng.multinomial(len(proteins), np.full(len(proteins), 1 / len(proteins)), size=r)
        weight = counts[:, index].astype(float)
        wpos, wneg = weight[:, pos], weight[:, neg]
        g = (weighted_contribution(ps_a, ns_a, wpos) - weighted_contribution(ps_b, ns_b, wpos))
        wo, wu = wneg * obs, wneg * ~obs
        so, su = wo.sum(1), wu.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            do = (wo * g).sum(1) / so
            du = np.where(su > 0, (wu * g).sum(1) / np.where(su > 0, su, 1), do)
            W = su / (so + su)
            out['c'].append(np.abs(du - do))
            out['c_star'].append(np.abs(do) / W)
            go, wk = g[:, obs], wneg[:, obs]
            for k in ks:
                eff, num = [], []
                for t in np.unique(tiers[k]):
                    sel = tiers[k] == t
                    nt = wk[:, sel].sum(1)
                    eff.append((wk[:, sel] * go[:, sel]).sum(1) / nt)
                    num.append(nt)
                eff, num = np.array(eff), np.array(num)
                dev = np.where(num > 0, np.abs(eff - do), -np.inf)   # empty tiers drop out
                out[f'h_{k}'].append(dev.max(axis=0))
                out[f'sd_{k}'].append(np.sqrt((num * np.nan_to_num(eff - do) ** 2).sum(0) / num.sum(0)))
        done += r
    res = {key: np.concatenate(v) for key, v in out.items()}
    for v in res.values():
        v[~np.isfinite(v)] = np.nan   # replicates that drew no observed-region protein
    return res


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
    for col in ['sd', 'c_star_over_sd', 'c_over_sd']:
        summary[col] = e.groupby('task', sort=False)[col].mean()
    summary.reset_index().to_csv(prefix + '_summary.tsv', sep='\t', index=False, float_format='%.6g')
    if args.reps:
        boot_rows = []
        rng = np.random.default_rng(args.seed)   # same stream as bootstrap_effects.py
        for task in TASKS:
            draws = []
            for split in range(args.splits):
                m, _ = paired_predictions(root, task, 'esm2', split, 'replica', 'rebuilt')
                dep = m.protein.map(depths[task]).fillna(0).to_numpy(int)
                draws.append(bootstrap_scale(m, dep, args.threshold, args.reps, rng))
            row = dict(task=task, reps=args.reps)
            for k in (2, 3, 4):
                for name, num, den in [('h', None, f'h_{k}'), ('c_over_h', 'c', f'h_{k}'),
                                       ('c_star_over_h', 'c_star', f'h_{k}'), ('sd', None, f'sd_{k}'),
                                       ('c_over_sd', 'c', f'sd_{k}'), ('c_star_over_sd', 'c_star', f'sd_{k}')]:
                    if num is None:
                        v = np.mean([d[den] for d in draws], axis=0)
                    else:   # mean over splits of split-specific ratios, as in the summary
                        v = np.mean([d[num] / d[den] for d in draws], axis=0)
                    lo, hi = np.nanpercentile(v, [2.5, 97.5])
                    row[f'{name}_{k}_lo'], row[f'{name}_{k}_hi'] = float(lo), float(hi)
            boot_rows.append(row)
            print(task, 'bootstrap complete', flush=True)
        pd.DataFrame(boot_rows).to_csv(prefix + '_bootstrap.tsv', sep='\t', index=False, float_format='%.6g')
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
    # Bootstrap: unit weights reproduce the point calibration, and the draws pair
    # with bootstrap_effects.bootstrap_pair under the same generator state.
    from bootstrap_effects import bootstrap_pair
    n = 400
    prot = np.array(['p%d' % (i % 37) for i in range(n)])
    pdepth = {f'p{i}': (i * 7) % 40 for i in range(37)}
    y = (rng.random(n) < .3).astype(int)
    m = pd.DataFrame({'protein': prot, 'y': y,
                      'y_pred_base': rng.random(n) + .4 * y, 'y_pred_aug': rng.random(n) + .5 * y})
    dep = np.array([pdepth[q] for q in prot])
    class Ones:
        def multinomial(self, n, p, size):
            return np.ones((size, len(p)), int)
    b = bootstrap_scale(m, dep, 10, 3, Ones(), ks=(3,))
    neg = y == 0
    g = (negative_contribution(m.y_pred_aug[~neg], m.y_pred_aug[neg])
         - negative_contribution(m.y_pred_base[~neg], m.y_pred_base[neg]))
    v, _ = calibrate(g, dep[neg], prot[neg], 10, 3)
    np.testing.assert_allclose([b['h_3'][0], b['sd_3'][0], b['c'][0], b['c_star'][0]],
                               [v['h'], v['sd'], v['c'], v['c_star']], rtol=1e-12)
    a = bootstrap_scale(m, dep, 10, 150, np.random.default_rng(5), ks=(3,))
    e = bootstrap_pair(m, dep, 10, 150, np.random.default_rng(5))
    np.testing.assert_allclose(a['c'], e['c']); np.testing.assert_allclose(a['c_star'], e['c_star'])
    print('calibrate_cstar self-tests: 5 passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='~/HRP')
    ap.add_argument('--root')
    ap.add_argument('--out', default='results_calibration')
    ap.add_argument('--splits', type=int, default=3)
    ap.add_argument('--threshold', type=int, default=10)
    ap.add_argument('--reps', type=int, default=2000, help='0 skips the bootstrap')
    ap.add_argument('--seed', type=int, default=20261004)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    self_test() if a.self_test else main(a)
