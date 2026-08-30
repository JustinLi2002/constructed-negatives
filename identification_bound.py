#!/usr/bin/env python3
"""Partial identification of the deployment effect under threshold negative
sampling.

A benchmark that draws negatives only from entities carrying at least T
positives supplies no negatives at all from entities below T.  The propensity of
appearing as a negative is exactly zero there, so this is a positivity
violation, not covariate shift: no reweighting of the benchmark identifies the
deployment-distribution effect.  What remains is partial identification.

The identity this rests on.  Hold the positive set fixed and let the negative
set be partitioned by the annotation depth of the entity each negative sits on,
N = union_k N_k.  AUROC is an average over positive-negative pairs, so

    AUROC = sum_k w_k * AUROC_k ,          w_k = |N_k| / |N|

and, differencing an augmented model against a baseline on the same pairs,

    delta_deploy = sum_k w_k * delta_k .

This is exact, not an approximation; --sweep reports the residual.  A benchmark
with threshold T supplies delta_k for strata with depth >= T and nothing below.
w_k needs only a candidate-item count per entity and that entity's annotation
depth -- no labels, no retraining, no access to the benchmark's negatives -- so
it is computable for any published benchmark from its release plus a reference
proteome.  That is what makes the framework usable on other people's papers.

Three estimators of the unobserved part are compared.

  naive   Assume a priori that delta_k is non-decreasing in depth.  Every
          unobserved stratum is then bounded above by the shallowest observed
          one.  This is the bound of the 2026-08-20 feasibility scan, and its
          failure mode is that the *direction* is assumed rather than read: when
          the harm mechanism has not engaged, shallow strata are better rather
          than worse and the bound is simply invalid.

  cond    Read the direction from the observed strata instead.  Two assumptions,
          both refutable inside the benchmark:
            (A) delta_k is monotone in depth, in the direction of the observed
                gradient;
            (B) the trend does not steepen outside the observed range -- the
                slope in log depth is bounded in magnitude by the observed one.
          Together these give a two-sided interval.  When the observed gradient
          rises with depth, (A) supplies the upper endpoint and (B) the lower;
          when it falls, they swap, and the upper bound comes from extrapolating
          the observed trend rather than from the endpoint.  That swap is what
          repairs the naive bound's failure at small T.

  oracle  The truth, computed from the unobserved strata.  Not available to an
          analyst; reported so validity can be checked.

Synthetic populations only.  The generative model carries the three structural
conditions of the first paper's synthetic_reversal.py and nothing else: labels
on items grouped into entities, negatives constructed by a donor-threshold rule,
and an entity-constant feature partially encoding how many positives the entity
has.  Truth is known by construction, which is the only way to check whether a
bound that cannot be checked on real data actually holds.

    python identification_bound.py --demo             # two cells, verbose
    python identification_bound.py --sweep            # the T grid

Provenance: the original script of this name was written 2026-08-20, produced
the feasibility table in HANDOVER_identification.md, and was never persisted --
it is not on the cluster, in the first paper's repository, or on this machine.
This is a re-implementation from that table's specification plus the first
paper's generator.  It is not seed-compatible with the original, so the numbers
it produces stand on their own; what should be checked against the original
table is the qualitative pattern, not the digits.
"""
import argparse

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Depth strata.  Bin i covers [EDGES[i], EDGES[i+1]).  The edges are placed so
# that every threshold under test is a bin boundary: a benchmark at T then
# observes exactly the strata with EDGES[i] >= T, and no stratum straddles the
# observed/unobserved line.
EDGES = [0, 1, 2, 3, 5, 10, 20, np.inf]
LABELS = ['0', '1', '2', '3-4', '5-9', '10-19', '20+']
THRESHOLDS = (0, 2, 5, 10, 20)


# --------------------------------- generation ------------------------------

def generate(n_entities=2500, feat_dim=16, local_dim=16, r2_target=0.30,
             prop_scale=1.15, rng=None):
    """One synthetic population, after the first paper's generator.

    Each entity carries a latent character z driving both its items' propensity
    to be positive and, through that, how many positives it ends up with.  The
    entity-constant feature encodes z with controllable fidelity, so it is
    genuinely informative about the label while also carrying the quantity the
    threshold rule turns into a sampling criterion.  Annotation depth is an
    observed consequence, not an input.
    """
    rng = rng or np.random.default_rng(0)

    n_items = 5 + rng.negative_binomial(3, 0.12, n_entities)
    z = rng.normal(0, 1, n_entities)
    prop = expit(-1.6 + prop_scale * z)

    w = rng.normal(0, 1, local_dim) / np.sqrt(local_dim)

    ent, x_local, y = [], [], []
    for e in range(n_entities):
        m = n_items[e]
        xi = rng.normal(0, 1, (m, local_dim))
        p = expit(xi @ w * 1.2 + logit(prop[e]))
        ent.append(np.full(m, e))
        x_local.append(xi)
        y.append(rng.random(m) < p)
    ent = np.concatenate(ent)
    x_local = np.vstack(x_local)
    y = np.concatenate(y).astype(int)

    depth = np.bincount(ent[y == 1], minlength=n_entities)

    direction = rng.normal(0, 1, feat_dim)
    direction /= np.linalg.norm(direction)
    feat = (np.sqrt(r2_target) * np.outer(z, direction)
            + np.sqrt(1 - r2_target) * rng.normal(0, 1, (n_entities, feat_dim)))

    return dict(ent=ent, x_local=x_local, y=y, depth=depth, feat=feat,
                n_entities=n_entities)


def construct(d, threshold, subset, ratio=5, rng=None):
    """Select positives and negatives under the sampling rule being studied.

    threshold = 0 draws negatives from every entity -- the deployment
    distribution a predictor actually meets.  threshold = T draws them only from
    entities carrying at least T positives, so entities below T contribute
    positives and exactly zero negatives.  The positive set does not depend on
    the threshold, which is what makes AUROC linear in the negative side.
    """
    rng = rng or np.random.default_rng(1)
    pos = np.where((d['y'] == 1) & subset)[0]
    neg_pool = np.where((d['y'] == 0) & subset)[0]
    if threshold > 0:
        donor = d['depth'] >= threshold
        neg_pool = neg_pool[donor[d['ent'][neg_pool]]]
    k = min(len(neg_pool), ratio * len(pos))
    neg = rng.choice(neg_pool, k, replace=False)
    return pos, neg


# --------------------------------- evaluation ------------------------------

CLASSIFIER = 'mlp'


def fit(d, tr_idx, augmented, seed=0):
    """Baseline (item-local features) or augmented (plus the entity feature).

    The default is the first paper's network.  A linear model is available for
    speed, but it does not reproduce the sign reversal: the entity channel has
    to be able to dominate the local one for the harm to appear at all, and with
    sixteen linear coefficients it cannot.  Under the linear model the truth
    stays positive through T = 10 and the bounds are being tested against a
    mechanism that is not the one under study.
    """
    def X(idx):
        if not augmented:
            return d['x_local'][idx]
        return np.hstack([d['x_local'][idx], d['feat'][d['ent'][idx]]])

    if CLASSIFIER == 'linear':
        est = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    else:
        est = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                            max_iter=400, early_stopping=True,
                            n_iter_no_change=12, random_state=seed)
    clf = make_pipeline(StandardScaler(), est)
    clf.fit(X(tr_idx), d['y'][tr_idx])
    return lambda idx: clf.predict_proba(X(idx))[:, 1]


def delta_against(pos_b, pos_a, neg_b, neg_a):
    """Augmented-minus-baseline AUROC of one fixed positive set against one
    negative set, both models scored on the same pairs."""
    yy = np.r_[np.ones(len(pos_b)), np.zeros(len(neg_b))]
    a = roc_auc_score(yy, np.r_[pos_a, neg_a])
    b = roc_auc_score(yy, np.r_[pos_b, neg_b])
    return a - b


def bin_of(depth_values):
    return np.digitize(depth_values, EDGES[1:-1], right=False)


# --------------------------------- the bounds ------------------------------

def naive_bound(w, delta, observed, anchor):
    """Assume delta_k non-decreasing in depth, a priori.  Every unobserved
    stratum then sits at or below the shallowest observed one."""
    inside = float(np.sum(w[observed] * delta[observed]))
    w_un = float(np.sum(w[~observed]))
    return inside + w_un * delta[anchor]


def observed_slope(w, delta, observed, mid):
    """Weighted least-squares slope of delta_k on log(1 + depth), over the
    strata the benchmark actually supplies."""
    if observed.sum() < 2:
        return 0.0
    lx = np.log1p(mid[observed])
    if np.ptp(lx) == 0:
        return 0.0
    ww = w[observed]
    if ww.sum() == 0:
        ww = np.ones(observed.sum())
    xm = np.average(lx, weights=ww)
    ym = np.average(delta[observed], weights=ww)
    den = np.sum(ww * (lx - xm) ** 2)
    return float(np.sum(ww * (lx - xm) * (delta[observed] - ym)) / den)


def extrapolate(w, delta, observed, mid, degree):
    """Point estimate of delta_deploy: the observed strata contribute exactly,
    the unobserved ones are predicted by a weighted polynomial in log(1 + depth)
    fitted on the observed strata alone and clipped to [-1, 1].

    degree 1 is the observed trend continued; degree 2 asks whether the
    curvature visible inside the observed range also predicts outside it, which
    is the question a shape restriction has to answer.  Returns (estimate,
    curvature) where curvature is the leading coefficient of the quadratic fit,
    positive when delta_k is convex in log depth.
    """
    inside = float(np.sum(w[observed] * delta[observed]))
    un = ~observed
    if not un.any():
        return inside, 0.0
    if observed.sum() <= degree:
        return np.nan, np.nan

    x = np.log1p(mid)
    sw = np.sqrt(np.maximum(w[observed], 1e-12))
    coef = np.polyfit(x[observed], delta[observed], degree, w=sw)
    pred = np.clip(np.polyval(coef, x[un]), -1.0, 1.0)
    return inside + float(np.sum(w[un] * pred)), float(coef[0])


def cond_bound(w, delta, observed, anchor, mid):
    """Direction read from the observed gradient, magnitude capped by it.

    With b the observed slope and u_k = log(1+mid_k) - log(1+mid_anchor), which
    is <= 0 for every unobserved stratum since all of them are shallower:

      b >= 0   monotonicity gives delta_k <= delta_anchor; the capped slope gives
               delta_k >= delta_anchor + b*u_k.  Upper endpoint-anchored, as in
               the naive bound, but the lower side is now informative too.
      b <  0   the two swap: delta_k >= delta_anchor, and delta_k <= delta_anchor
               + b*u_k, the observed trend extrapolated into the unobserved
               region.  The upper bound no longer comes from the endpoint, which
               is what repairs the naive bound where it was invalid.

    Both endpoints are clipped to [-1, 1], the range of a difference of AUROCs.
    """
    inside = float(np.sum(w[observed] * delta[observed]))
    un = ~observed
    if not un.any():
        return inside, inside, 0.0
    if observed.sum() < 2:
        # no gradient is estimable, so neither assumption says anything: the
        # honest interval is the whole range a difference of AUROCs can take.
        w_un = float(np.sum(w[un]))
        return inside - w_un, inside + w_un, np.nan

    b = observed_slope(w, delta, observed, mid)
    u = np.log1p(mid[un]) - np.log1p(mid[anchor])
    extrap = np.clip(delta[anchor] + b * u, -1.0, 1.0)
    flat = np.full(int(un.sum()), delta[anchor])

    lo_k, hi_k = (extrap, flat) if b >= 0 else (flat, extrap)
    return (inside + float(np.sum(w[un] * lo_k)),
            inside + float(np.sum(w[un] * hi_k)), b)


def monotonicity_violation(delta, observed):
    """Largest amount by which a deeper observed stratum falls below a shallower
    one -- how badly the a priori assumption fails where it can be checked."""
    dk = delta[observed]
    worst = 0.0
    for i in range(len(dk)):
        for j in range(i + 1, len(dk)):
            worst = max(worst, dk[i] - dk[j])
    return float(worst)


# --------------------------------- one cell --------------------------------

def one_cell(threshold, seed=0, n_entities=2500, r2_target=0.30, verbose=False):
    rng = np.random.default_rng(seed)
    d = generate(n_entities=n_entities, r2_target=r2_target, rng=rng)

    perm = rng.permutation(d['n_entities'])
    test_ent = np.zeros(d['n_entities'], bool)
    test_ent[perm[:d['n_entities'] // 5]] = True
    in_test = test_ent[d['ent']]

    # the model an analyst using this benchmark would have: trained on the
    # threshold-constructed training split
    tr_pos, tr_neg = construct(d, threshold, ~in_test, rng=rng)
    tr = np.concatenate([tr_pos, tr_neg])
    score_b = fit(d, tr, False, seed)
    score_a = fit(d, tr, True, seed)

    # two test partitions over the SAME held-out entities and the SAME positive
    # set, differing only in how their negatives were drawn
    pos, neg_deploy = construct(d, 0, in_test, rng=np.random.default_rng(seed + 99))
    _, neg_bench = construct(d, threshold, in_test,
                             rng=np.random.default_rng(seed + 99))

    pb, pa = score_b(pos), score_a(pos)
    d_bench = delta_against(pb, pa, score_b(neg_bench), score_a(neg_bench))
    d_deploy = delta_against(pb, pa, score_b(neg_deploy), score_a(neg_deploy))

    # ---- exact stratum decomposition over the deployment negatives ----
    K = len(LABELS)
    dep_depth = d['depth'][d['ent'][neg_deploy]]
    nb = bin_of(dep_depth)
    sb_d, sa_d = score_b(neg_deploy), score_a(neg_deploy)
    w = np.zeros(K)
    delta_true = np.zeros(K)
    mid = np.zeros(K)
    for k in range(K):
        m = nb == k
        w[k] = m.sum() / len(neg_deploy)
        if m.sum():
            mid[k] = dep_depth[m].mean()
            delta_true[k] = delta_against(pb, pa, sb_d[m], sa_d[m])
    resid = abs(float(np.sum(w * delta_true)) - d_deploy)

    # ---- what the benchmark actually supplies ----
    # delta_k estimated from the BENCHMARK's own negatives, since that is what an
    # analyst has; w_k from the deployment universe, which needs only depths and
    # candidate counts and so is computable without the benchmark.
    nbb = bin_of(d['depth'][d['ent'][neg_bench]])
    sb_b, sa_b = score_b(neg_bench), score_a(neg_bench)
    delta_obs = np.zeros(K)
    have = np.zeros(K, bool)
    for k in range(K):
        m = nbb == k
        if m.sum() >= 5:
            delta_obs[k] = delta_against(pb, pa, sb_b[m], sa_b[m])
            have[k] = True

    observed = np.array([EDGES[k] >= threshold
                         for k in range(K)]) & have & (w > 0)
    if not observed.any():
        return None
    anchor = int(np.where(observed)[0][0])

    nb_ = naive_bound(w, delta_obs, observed, anchor)
    lo, hi, slope = cond_bound(w, delta_obs, observed, anchor, mid)
    lin, _ = extrapolate(w, delta_obs, observed, mid, 1)
    quad, curv = extrapolate(w, delta_obs, observed, mid, 2)

    row = dict(threshold=threshold, seed=seed,
               bench=d_bench, truth=d_deploy, resid=resid,
               naive=nb_, cond_lo=lo, cond_hi=hi, slope=slope,
               lin=lin, quad=quad, curv=curv,
               lin_err=abs(lin - d_deploy), quad_err=abs(quad - d_deploy),
               naive_err=abs(nb_ - d_deploy),
               lin_upper=int(d_deploy <= lin + 1e-12),
               quad_upper=int(d_deploy <= quad + 1e-12),
               w_unobs=float(np.sum(w[~observed])),
               n_obs=int(observed.sum()),
               viol=monotonicity_violation(delta_obs, observed),
               naive_valid=int(d_deploy <= nb_ + 1e-12),
               cond_valid=int(lo - 1e-12 <= d_deploy <= hi + 1e-12),
               naive_excl_bench=int(nb_ < d_bench),
               cond_excl_bench=int(hi < d_bench),
               naive_excl_zero=int(nb_ < 0),
               cond_excl_zero=int(hi < 0))

    if verbose:
        print('\n  T = %d, seed %d' % (threshold, seed))
        print('  %8s %7s %13s %12s  %s'
              % ('stratum', 'w_k', 'delta_k true', 'delta_k obs', 'status'))
        for k in range(K):
            if w[k] == 0 and not have[k]:
                continue
            st = 'observed' if observed[k] else 'UNOBSERVED'
            o = '%+.4f' % delta_obs[k] if observed[k] else '--'
            print('  %8s %7.3f %+13.4f %12s  %s'
                  % (LABELS[k], w[k], delta_true[k], o, st))
        print('  benchmark reads %+.4f   truth %+.4f' % (d_bench, d_deploy))
        print('  naive upper %+.4f   cond [%+.4f, %+.4f]   slope %+.4f'
              % (nb_, lo, hi, slope))
        print('  decomposition residual %.2e' % resid)
    return row


# --------------------------------- drivers ---------------------------------

def mean_of(rows, key):
    v = np.asarray([r[key] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def nan_or(fmt, v, alt='--'):
    return alt.rjust(len(fmt % 0.0)) if not np.isfinite(v) else fmt % v


def sweep(args):
    print('identification_bound.py -- %d seeds x %d entities per cell, %s\n'
          % (args.seeds, args.n_entities, CLASSIFIER))
    allrows = []
    rowsby = {}

    head = ('%3s %7s %8s %8s | %8s %6s | %8s %8s %6s %7s | %5s %5s'
            % ('T', 'w_unob', 'bench', 'truth', 'naive', 'holds',
               'cond lo', 'cond hi', 'holds', 'slope', 'viol', 'n_obs'))
    print('the bound, as it stands')
    print(head)
    print('-' * len(head))
    for t in THRESHOLDS:
        rows = [r for r in (one_cell(t, seed=s, n_entities=args.n_entities)
                            for s in range(args.seeds)) if r]
        allrows += rows
        rowsby[t] = rows
        print('%3d %7.2f %+8.3f %+8.3f | %+8.3f %5.0f%% | %+8.3f %+8.3f '
              '%5.0f%% %s | %5.2f %5.1f'
              % (t, mean_of(rows, 'w_unobs'), mean_of(rows, 'bench'),
                 mean_of(rows, 'truth'),
                 mean_of(rows, 'naive'), 100 * mean_of(rows, 'naive_valid'),
                 mean_of(rows, 'cond_lo'), mean_of(rows, 'cond_hi'),
                 100 * mean_of(rows, 'cond_valid'),
                 nan_or('%+7.3f', mean_of(rows, 'slope')),
                 mean_of(rows, 'viol'), mean_of(rows, 'n_obs')), flush=True)
    print('-' * len(head))

    head2 = ('%3s %8s | %8s %6s %6s | %8s %6s %6s | %8s %6s %6s | %7s'
             % ('T', 'truth', 'flat', 'holds', 'err', 'linear', 'holds', 'err',
                'quadratic', 'holds', 'err', 'curv'))
    print('\nhow the unobserved region is filled in: endpoint, observed trend '
          'continued,\nobserved curvature continued.  holds = share of '
          'replicates where the estimate is\nan upper bound on the truth; err '
          '= mean |estimate - truth|.')
    print(head2)
    print('-' * len(head2))
    for t in THRESHOLDS:
        rows = rowsby[t]
        cells = []
        for est, hold, err in (('naive', 'naive_valid', 'naive_err'),
                               ('lin', 'lin_upper', 'lin_err'),
                               ('quad', 'quad_upper', 'quad_err')):
            cells += [nan_or('%+8.3f', mean_of(rows, est)),
                      nan_or('%5.0f', 100 * mean_of(rows, hold)) + '%',
                      nan_or('%6.3f', mean_of(rows, err))]
        print('%3d %+8.3f | %s %s %s | %s %s %s | %s %s %s | %s'
              % (t, mean_of(rows, 'truth'), *cells,
                 nan_or('%+7.3f', mean_of(rows, 'curv'))), flush=True)
    print('-' * len(head2))

    print('\nmax decomposition residual over all cells: %.2e'
          % max(r['resid'] for r in allrows))
    if args.out:
        import csv
        with open(args.out, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(allrows[0]), delimiter='\t')
            wr.writeheader()
            wr.writerows(allrows)
        print('per-replicate rows written to %s' % args.out)


def demo(args):
    for t in (2, 10):
        one_cell(t, seed=0, n_entities=args.n_entities, verbose=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--n-entities', type=int, default=2500)
    ap.add_argument('--out', default='')
    ap.add_argument('--clf', default='mlp', choices=['mlp', 'linear'])
    a = ap.parse_args()
    CLASSIFIER = a.clf
    globals()['CLASSIFIER'] = a.clf
    if a.demo:
        demo(a)
    else:
        sweep(a)
