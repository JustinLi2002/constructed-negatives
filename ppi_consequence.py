#!/usr/bin/env python3
"""Does the PPI positivity violation actually change a conclusion?

The taxonomy establishes that the configuration-model prefilter leaves an exact
zero above a degree cutoff.  That is a statement about sampling geometry.  It is
not yet a statement that anyone's conclusion is wrong, and a reviewer is
entitled to ask for the second one.  This supplies it, in the same shape as the
first paper's result for PTM: a node-level feature that appears to help when
scored against the benchmark's negatives and in fact harms on the distribution a
predictor meets, with every diagnostic computable inside the benchmark clean.

Design: **real topology, real rule, controlled features.**  The network, its
degree distribution and the sampling rule are taken from the real case, so the
geometry that drives the result is not invented.  The features are constructed,
so ground truth is known and the effect is attributable to the sampling rule
rather than to some idiosyncrasy of an embedding.  That is the same bargain the
first paper's synthetic_reversal.py strikes, with the graph made real.

  positives          STRING physical edges, held fixed across both evaluations
  benchmark negs     non-edges admitted by the configuration-model prefilter,
                     d_u * d_v <= tau -- so every negative sits on a pair of
                     low-degree proteins and nothing above the cutoff appears
  deployment negs    non-edges drawn uniformly, which is what a predictor is
                     applied to
  baseline           a pair-local channel standing in for evidence about the
                     specific interaction, calibrated to a target AUROC
  augmented          the same, plus a protein-level channel encoding log degree
                     with fidelity r2 -- the analogue of an embedding that
                     reflects how well studied a protein is

The split is protein-disjoint: test pairs join two proteins neither of which
appears in any training pair, so nothing here is a leakage effect.

    python ppi_consequence.py data/string_phys_700.tsv --seeds 5
"""
import argparse

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cl3_exclusion import config_threshold, load_edges

# Deployment negatives are stratified by the pair's degree product, because that
# is the quantity the prefilter cuts on: the benchmark supplies strata at or
# below tau and nothing above it.
PROD_EDGES = [1, 2, 3, 5, 10, 25, 100, 1000, np.inf]
PROD_LABELS = ['1', '2', '3-4', '5-9', '10-24', '25-99', '100-999', '1000+']


def sample_nonedges(edge_set, allowed, n_want, rng, predicate=None,
                    max_draws=400):
    """Draw distinct non-adjacent pairs from `allowed`, optionally filtered."""
    out, seen = [], set()
    m = len(allowed)
    if m < 2:
        return np.zeros((0, 2), np.int64)
    for _ in range(max_draws * max(n_want, 1)):
        if len(out) >= n_want:
            break
        u = allowed[rng.integers(m)]
        v = allowed[rng.integers(m)]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in edge_set or key in seen:
            continue
        if predicate is not None and not predicate(key[0], key[1]):
            continue
        seen.add(key)
        out.append(key)
    return np.array(out, dtype=np.int64)


def make_features(pairs, y, deg, node_feat, rng, sep=0.55, loc_dim=8):
    """Pair-local channel (calibrated to the label) and protein-level channel.

    The local channel is drawn from the label alone, identically for benchmark
    and deployment negatives, so it carries no information about which sampling
    rule produced a pair -- any difference between the two evaluations comes
    from the negatives themselves, not from the features.
    """
    n = len(pairs)
    e = np.zeros(loc_dim)
    e[0] = 1.0
    x_local = rng.normal(0, 1, (n, loc_dim)) + sep * np.outer(y, e)

    fu, fv = node_feat[pairs[:, 0]], node_feat[pairs[:, 1]]
    x_node = np.hstack([fu * fv, np.abs(fu - fv)])   # symmetric in the pair
    return x_local, x_node


def fit_model(x_tr, y_tr, seed):
    """Fit once; the two evaluations differ only in their negatives, so they
    share a training set and there is no reason to fit twice for them.  Scoring
    a single fitted model on both is identical to fitting twice with the same
    random_state, and halves the cost -- which matters on the dense networks,
    where the training set is six times larger than STRING's."""
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                      max_iter=400, early_stopping=True,
                                      n_iter_no_change=12, random_state=seed))
    clf.fit(x_tr, y_tr)
    return clf


# Cap on training pairs.  The quantity under study is a property of the negative
# *distribution*, not of sample size, and eight to twenty-four features do not
# need more than this to fit.  Without a cap the cost scales with network
# density: STRING trains on ~110k pairs and HIPPIE on ~640k, which is the
# difference between a task that finishes in an hour and one that does not
# finish at all.  Set above STRING's requirement so those results are unchanged.
MAX_TRAIN = 200_000


def one_run(A, deg, tau, seed, r2=0.30, feat_dim=8, ratio=1):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    U = sparse.triu(A, k=1).tocoo()
    edge_set = set(zip(U.row.tolist(), U.col.tolist()))

    # protein-level channel: encodes log degree with fidelity r2, exactly as the
    # first paper's entity feature encodes annotation depth
    t = np.log1p(deg).astype(float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    direction = rng.normal(0, 1, feat_dim)
    direction /= np.linalg.norm(direction)
    node_feat = (np.sqrt(r2) * np.outer(t, direction)
                 + np.sqrt(1 - r2) * rng.normal(0, 1, (n, feat_dim)))

    # protein-disjoint split
    perm = rng.permutation(n)
    is_test = np.zeros(n, bool)
    is_test[perm[:n // 5]] = True
    tr_nodes = np.where(~is_test)[0]
    te_nodes = np.where(is_test)[0]

    def edges_within(mask):
        m = mask[U.row] & mask[U.col]
        return np.stack([U.row[m], U.col[m]], axis=1).astype(np.int64)

    pos_tr = edges_within(~is_test)
    pos_te = edges_within(is_test)
    if len(pos_te) < 200 or len(pos_tr) < 200:
        return None

    admits = lambda u, v: deg[u] * deg[v] <= tau

    # cap before drawing, not after: the rejection sampler is the second cost
    # here and there is no point drawing negatives that get thrown away
    keep = MAX_TRAIN // (1 + ratio)
    if len(pos_tr) > keep:
        pos_tr = pos_tr[rng.choice(len(pos_tr), keep, replace=False)]

    neg_tr = sample_nonedges(edge_set, tr_nodes, ratio * len(pos_tr), rng,
                             predicate=admits)
    neg_bench = sample_nonedges(edge_set, te_nodes, ratio * len(pos_te), rng,
                                predicate=admits)
    neg_deploy = sample_nonedges(edge_set, te_nodes, ratio * len(pos_te), rng)
    if min(len(neg_tr), len(neg_bench), len(neg_deploy)) < 100:
        return None

    def build(pos, neg):
        pairs = np.vstack([pos, neg])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        xl, xn = make_features(pairs, y, deg, node_feat, rng)
        return pairs, y, xl, xn

    _, y_tr, xl_tr, xn_tr = build(pos_tr, neg_tr)
    pr_b, y_b, xl_b, xn_b = build(pos_te, neg_bench)
    pr_d, y_d, xl_d, xn_d = build(pos_te, neg_deploy)

    out = {'seed': seed, 'tau': int(tau),
           'n_pos_te': len(pos_te), 'n_neg_bench': len(neg_bench)}

    m_base = fit_model(xl_tr, y_tr, seed)
    m_aug = fit_model(np.hstack([xl_tr, xn_tr]), y_tr, seed)

    for tag, (xl_te, xn_te, y_te) in (('bench', (xl_b, xn_b, y_b)),
                                      ('deploy', (xl_d, xn_d, y_d))):
        base = m_base.predict_proba(xl_te)[:, 1]
        aug = m_aug.predict_proba(np.hstack([xl_te, xn_te]))[:, 1]
        out['auroc_base_' + tag] = roc_auc_score(y_te, base)
        out['auroc_aug_' + tag] = roc_auc_score(y_te, aug)
        out['delta_' + tag] = out['auroc_aug_' + tag] - out['auroc_base_' + tag]
        if tag == 'deploy':
            out['_scores'] = (base, aug, y_d, pr_d)
    return out


def decompose(out, deg):
    """Exact stratum decomposition of the deployment effect, by degree product."""
    base, aug, y, pairs = out.pop('_scores')
    pos = y == 1
    neg = ~pos
    prod = deg[pairs[neg, 0]] * deg[pairs[neg, 1]]
    b = np.digitize(prod, PROD_EDGES[1:-1], right=False)

    # w_unobs from the rule's own admissibility, not from the bin edges.  The
    # bins are fixed and top out at 1000, so on a dense network where tau lands
    # above that edge the whole top bin gets marked observed and the reported
    # weight collapses to zero -- HIPPIE at frac = 0.5 has tau = 1680 and read
    # 0.000.  The effect values were never affected, since those are computed on
    # the sampled sets rather than the bins, but the annotation was wrong.
    w_unobs = float((prod > out['tau']).mean())

    rows, tot = [], 0.0
    for k in range(len(PROD_LABELS)):
        m = b == k
        if not m.any():
            continue
        w = m.sum() / neg.sum()
        yy = np.r_[np.ones(pos.sum()), np.zeros(m.sum())]
        da = roc_auc_score(yy, np.r_[aug[pos], aug[neg][m]])
        db = roc_auc_score(yy, np.r_[base[pos], base[neg][m]])
        rows.append((PROD_LABELS[k], w, da - db, PROD_EDGES[k] <= out['tau']))
        tot += w * (da - db)
    return rows, tot, w_unobs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('edges')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--prefilter-frac', type=float, default=0.064)
    ap.add_argument('--r2', type=float, default=0.30)
    ap.add_argument('--sweep', action='store_true',
                    help='map the effect over the prefilter-severity x feature-'
                         'fidelity plane, the PPI analogue of the first '
                         "paper's T x R^2 figure")
    ap.add_argument('--out', default='ppi_sweep')
    ap.add_argument('--max-train', type=int, default=None,
                    help='override the training-pair cap; for testing how much '
                         'the cap costs')
    ap.add_argument('--only-frac', type=float, default=None,
                    help='run a single prefilter severity, so a sweep can be '
                         'split across array tasks')
    a = ap.parse_args()
    if a.max_train:
        MAX_TRAIN = a.max_train
        globals()['MAX_TRAIN'] = a.max_train

    A, names = load_edges(a.edges)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    tau = config_threshold(deg, a.prefilter_frac)
    print('%d proteins, %d edges, configuration-model cutoff tau = %d'
          % (A.shape[0], A.nnz // 2, tau))
    print('proteins above the cutoff (degree > tau): %d of %d, carrying %.3f '
          'of all degree\n'
          % (int((deg > tau).sum()), len(deg),
             deg[deg > tau].sum() / deg.sum()))

    if a.sweep:
        # frac = 1.0 keeps every non-edge, i.e. random sampling: the control
        fracs = [a.only_frac] if a.only_frac else [0.064, 0.2, 0.5, 1.0]
        r2s = [0.05, 0.15, 0.30, 0.60, 0.90]
        print('%8s %6s %6s | %9s %10s %9s %8s'
              % ('frac', 'tau', 'r2', 'benchmark', 'deployment', 'w_unobs',
                 'flips'))
        print('-' * 70)
        rec = []
        for f in fracs:
            tf = config_threshold(deg, f) if f < 1.0 else int(deg.max()) ** 2
            for r2 in r2s:
                rs = [one_run(A, deg, tf, sd, r2=r2) for sd in range(a.seeds)]
                rs = [r for r in rs if r]
                if not rs:
                    continue
                wun = []
                for r in rs:
                    _, _, wu = decompose(r, deg)
                    wun.append(wu)
                db = float(np.mean([r['delta_bench'] for r in rs]))
                dd = float(np.mean([r['delta_deploy'] for r in rs]))
                fl = sum(1 for r in rs
                         if r['delta_bench'] > 0 > r['delta_deploy'])
                rec.append((f, tf, r2, db, dd, float(np.mean(wun)), fl, len(rs)))
                print('%8.3f %6d %6.2f | %+9.4f %+10.4f %9.3f %4d/%d'
                      % (f, tf, r2, db, dd, np.mean(wun), fl, len(rs)),
                      flush=True)
        cols = ['prefilter_frac', 'tau', 'r2', 'delta_bench', 'delta_deploy',
                'w_unobs', 'n_flip', 'n_seed']
        fmt = '%.3f\t%d\t%.2f\t%.6f\t%.6f\t%.4f\t%d\t%d\n'
        with open(a.out + '.tsv', 'w', encoding='utf-8') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in rec:
                fh.write(fmt % r)
        print('\nwritten: %s.tsv' % a.out)
        raise SystemExit

    runs, last = [], None
    for s in range(a.seeds):
        r = one_run(A, deg, tau, s, r2=a.r2)
        if r is None:
            continue
        last = decompose(r, deg)
        runs.append(r)
        print('  seed %d  benchmark %+0.4f   deployment %+0.4f'
              % (s, r['delta_bench'], r['delta_deploy']), flush=True)

    if not runs:
        raise SystemExit('no usable replicates')

    m = lambda k: float(np.mean([r[k] for r in runs]))
    print('\n%-34s %10s %10s' % ('', 'benchmark', 'deployment'))
    print('  ' + '-' * 54)
    for lab, k in (('AUROC, local channel only', 'auroc_base_'),
                   ('AUROC, plus protein channel', 'auroc_aug_'),
                   ('difference', 'delta_')):
        print('%-34s %10.4f %10.4f' % (lab, m(k + 'bench'), m(k + 'deploy')))
    print('  ' + '-' * 54)
    sign_flip = sum(1 for r in runs
                    if r['delta_bench'] > 0 > r['delta_deploy'])
    print('replicates where the benchmark says help and deployment says harm:'
          ' %d / %d' % (sign_flip, len(runs)))

    rows, tot, w_un_exact = last
    print('\nstratum decomposition of the deployment effect (last replicate),'
          '\nnegatives binned by the pair degree product the prefilter cuts on:')
    print('  %10s %8s %12s   %s' % ('product', 'w_k', 'delta_k', 'benchmark'))
    print('  ' + '-' * 52)
    for lab, w, d, obs in rows:
        print('  %10s %8.3f %+12.4f   %s'
              % (lab, w, d, 'supplies this' if obs else 'SUPPLIES NOTHING'))
    print('  ' + '-' * 52)
    print('  sum of w_k * delta_k = %+0.4f  (deployment delta %+0.4f)'
          % (tot, runs[-1]['delta_deploy']))
    print('  weight the benchmark supplies nothing for: %.3f  '
          '(binned: %.3f)'
          % (w_un_exact, sum(w for _, w, _, obs in rows if not obs)))
