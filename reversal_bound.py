#!/usr/bin/env python3
"""What the unobserved weight guarantees, with no shape restriction at all.

The sweep showed the benchmark's error growing monotonically with the unobserved
weight.  That relationship is not a trend to be fitted; it is an identity, and
writing it down turns a plot into a proposition.

Let the deployment negatives be partitioned by stratum, let O be the strata the
benchmark supplies and U those it does not, and let

    W_un = sum_{k in U} w_k ,
    delta_O    = (1 / (1 - W_un)) * sum_{k in O} w_k delta_k ,
    delta_bar  = (1 / W_un)       * sum_{k in U} w_k delta_k .

delta_O is what a benchmark restricted to O estimates: the effect on its own
negative distribution.  delta_bar is the average effect over everything it never
sees.  Then, from delta_deploy = sum_k w_k delta_k alone,

    delta_deploy - delta_O = W_un * (delta_bar - delta_O)                   (1)

exactly.  The error is the unobserved weight times the gap between what is out
there and what was measured.  Nothing is assumed; (1) is arithmetic.

The corollary is the useful part.  A sign reversal means delta_O > 0 >
delta_deploy.  Substituting (1) and using only delta_bar >= -1, which holds
because delta_bar is an average of differences of AUROCs,

    a reversal requires   W_un  >  delta_O / (delta_O + 1) .                (2)

So if W_un <= delta_O / (delta_O + 1), no sign reversal is possible whatever the
unobserved region contains.  Both sides are computable inside the benchmark --
delta_O is its own reading, W_un needs only entity counts and depths -- and the
guarantee costs no shape restriction, which is the thing the naive bound needed
and could not defend.  It is a one-sided guarantee: above the threshold a
reversal becomes possible, not certain.

What this does not do is bound delta_deploy tightly.  (2) is the worst case over
an unobserved region that could be anything, so it is permissive by
construction.  That is the honest division of labour: this says when you are
safe without assumptions, and a shape restriction is still what you need to say
where the truth lies when you are not.

    python reversal_bound.py data/string_phys_700.tsv
"""
import argparse

import numpy as np

from cl3_exclusion import config_threshold, load_edges
from ppi_consequence import PROD_EDGES, PROD_LABELS, one_run

from sklearn.metrics import roc_auc_score


def split_effect(out, deg, tau):
    """Compute W_un, delta_O and delta_bar from one run's deployment scores.

    The split is by the rule's own admissibility -- a deployment negative sits
    in O exactly when the benchmark's sampler could have drawn it -- so this is
    the partition the benchmark actually induces, not a binning choice.
    """
    base, aug, y, pairs = out['_scores']
    pos = y == 1
    neg = ~pos
    npair = pairs[neg]
    prod = deg[npair[:, 0]] * deg[npair[:, 1]]
    in_O = prod <= tau

    def delta_on(mask):
        if mask.sum() == 0:
            return np.nan
        yy = np.r_[np.ones(pos.sum()), np.zeros(mask.sum())]
        return (roc_auc_score(yy, np.r_[aug[pos], aug[neg][mask]])
                - roc_auc_score(yy, np.r_[base[pos], base[neg][mask]]))

    w_un = float((~in_O).mean())
    return w_un, delta_on(in_O), delta_on(~in_O), delta_on(np.ones(len(npair), bool))


def main(path, seeds, fracs, r2s):
    A, names = load_edges(path)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    print('%d proteins, %d edges\n' % (A.shape[0], A.nnz // 2))

    print('%6s %7s %6s | %8s %9s %8s | %9s %9s | %s'
          % ('frac', 'W_un', 'r2', 'delta_O', 'delta_bar', 'deploy',
             'identity', 'threshold', 'verdict'))
    print('-' * 92)

    worst = 0.0
    rows = []
    for f in fracs:
        tau = config_threshold(deg, f) if f < 1.0 else int(deg.max()) ** 2
        for r2 in r2s:
            acc = []
            for s in range(seeds):
                r = one_run(A, deg, tau, s, r2=r2)
                if r is None:
                    continue
                acc.append(split_effect(r, deg, tau))
            if not acc:
                continue
            w_un = float(np.mean([a[0] for a in acc]))
            d_O = float(np.nanmean([a[1] for a in acc]))
            _bar = [a[2] for a in acc if np.isfinite(a[2])]
            d_bar = float(np.mean(_bar)) if _bar else np.nan
            d_dep = float(np.nanmean([a[3] for a in acc]))

            # identity (1), checked per replicate then averaged
            res = [abs((a[3] - a[1]) - a[0] * (a[2] - a[1]))
                   for a in acc if np.isfinite(a[1]) and np.isfinite(a[2])]
            res = float(np.mean(res)) if res else np.nan
            worst = max(worst, res if np.isfinite(res) else 0.0)

            # corollary (2): the largest W_un that still forbids a reversal
            thr = d_O / (d_O + 1) if d_O > 0 else np.nan
            if not np.isfinite(thr):
                verdict = 'n/a (delta_O <= 0)'
            elif w_un <= thr:
                verdict = 'reversal IMPOSSIBLE'
            else:
                verdict = 'reversal possible'
            flipped = d_O > 0 > d_dep
            if flipped:
                verdict += ' -- and it happened'

            rows.append((f, w_un, r2, d_O, d_bar, d_dep, res, thr, verdict,
                         flipped))
            print('%6.3f %7.3f %6.2f | %+8.4f %+9.4f %+8.4f | %9.2e %9s | %s'
                  % (f, w_un, r2, d_O, d_bar, d_dep, res,
                     ('%.3f' % thr) if np.isfinite(thr) else '--', verdict),
                  flush=True)

    print('-' * 92)
    print('worst residual on identity (1): %.2e' % worst)

    viol = [r for r in rows if r[9] and np.isfinite(r[7]) and r[1] <= r[7]]
    print('cells where a reversal occurred despite the guarantee: %d'
          % len(viol))
    if viol:
        print('  ** the corollary is wrong as stated -- investigate **')
        for r in viol:
            print('   frac %.3f r2 %.2f  W_un %.3f  threshold %.3f'
                  % (r[0], r[2], r[1], r[7]))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('edges')
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    main(a.edges, a.seeds,
         fracs=[0.064, 0.2, 0.5, 1.0],
         r2s=[0.15, 0.30, 0.60, 0.90])
