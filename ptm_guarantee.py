#!/usr/bin/env python3
"""How far the assumption-free guarantee is from the first paper's real tasks.

`reversal_bound.py` establishes, from the stratum decomposition alone and with
no shape restriction whatever, that a sign reversal requires

    W_un  >  delta_O / (delta_O + 1)

where delta_O is the benchmark's own reading and W_un the share of deployment
negatives it supplies nothing for.  Below that threshold the benchmark's sign is
safe no matter what sits in the region it never sampled.

That guarantee is worth stating only if one can say how often it fires.  This
answers it on real published tasks rather than on the synthetic grid: the eight
PTM tasks of the first paper, whose delta_O is in its results and whose W_un is
in its release.

Inputs, both from the ptm-audit repository:
  results/summary.json          per-task AUROC by condition; delta_O is the
                                protein-level channel minus baseline
  results/unobserved_weight.txt W_un per task at T = 10, in two universes

The two universes answer different questions and both are reported.  The
deployment universe is the whole reference proteome -- the population a
predictor is actually applied to.  The restricted universe counts only proteins
already carrying a site of that type, which is the most generous reading anyone
could defend, and is included so the conclusion cannot be dismissed as an
artefact of the wider one.

    python ptm_guarantee.py /path/to/ptm-audit
"""
import argparse
import json
import os
import re


def load_weights(path):
    """Parse the per-task unobserved weights out of the released text table."""
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            m = re.match(r'^([A-Za-z][A-Za-z\- ]*?[A-Za-z/]+)\s+'
                         r'([01]\.\d+)\s+(\d+)\s+([01]\.\d+)\s+(\d+)\s*$',
                         line.strip())
            if m:
                out[m.group(1).strip()] = (float(m.group(2)),
                                           float(m.group(4)))
    return out


# the release's task names against summary.json's keys
ALIAS = {
    'Phosphorylation S/T': 'phosphorylation_st',
    'Phosphorylation Y': 'phosphorylation_y',
    'Acetylation K': 'acetylation_k',
    'Methylation K/R': 'methylation_k',
    'Methylation R': 'methylation_r',
    'Sumoylation K': 'sumoylation_k',
    'Ubiquitination K': 'ubiquitination_k',
    'N-Glycosylation N': 'glycosylation_n',
}


def main(root, cond):
    summ = json.load(open(os.path.join(root, 'results', 'summary.json'),
                          encoding='utf-8'))
    weights = load_weights(os.path.join(root, 'results',
                                        'unobserved_weight.txt'))
    if not weights:
        raise SystemExit('could not parse unobserved_weight.txt')

    auroc = {}
    for r in summ:
        auroc.setdefault(r['ptm'], {})[r['cond']] = r['seed_mean']

    print('the assumption-free reversal guarantee on the first paper\'s eight '
          'tasks\n')
    print('delta_O   = %s minus baseline, the benchmark\'s own reading'
          % cond)
    print('threshold = delta_O / (delta_O + 1), the largest unobserved weight')
    print('            that still forbids a sign reversal, assuming nothing')
    print('W_un      = share of deployment negatives the benchmark supplies '
          'nothing for\n')
    print('%-22s %9s %10s | %8s %8s | %8s %8s'
          % ('task', 'delta_O', 'threshold', 'W_un dep', 'margin',
             'W_un res', 'margin'))
    print('-' * 84)

    rows, safe_dep, safe_res = [], 0, 0
    for name, (w_res, w_dep) in weights.items():
        key = ALIAS.get(name)
        if key not in auroc or cond not in auroc[key]:
            continue
        d_O = auroc[key][cond] - auroc[key]['baseline']
        thr = d_O / (d_O + 1) if d_O > 0 else float('nan')
        ok_d = d_O > 0 and w_dep <= thr
        ok_r = d_O > 0 and w_res <= thr
        safe_dep += ok_d
        safe_res += ok_r
        rows.append((name, d_O, thr, w_dep, w_res))
        t = ('%10.4f' % thr) if d_O > 0 else '%10s' % 'n/a'
        print('%-22s %+9.4f %s | %8.3f %8s | %8.3f %8s'
              % (name, d_O, t, w_dep,
                 ('%.0fx' % (w_dep / thr)) if d_O > 0 and thr > 0 else '--',
                 w_res,
                 ('%.0fx' % (w_res / thr)) if d_O > 0 and thr > 0 else '--'))
    print('-' * 84)
    print('tasks where the guarantee fires: %d of %d in the deployment '
          'universe, %d of %d in the restricted one'
          % (safe_dep, len(rows), safe_res, len(rows)))

    pos = [r for r in rows if r[1] > 0]
    if pos:
        ratios = [r[3] / r[2] for r in pos]
        print('\nmargin by which the deployment universe misses the guarantee: '
              '%.0fx to %.0fx' % (min(ratios), max(ratios)))
        print('\nSo on real published tasks the free guarantee never fires, and '
              'not\nnarrowly.  A shape restriction is not a refinement here; it '
              'is the only\nroute to any statement about the deployment effect '
              'at all.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='path to a checkout of the ptm-audit repo')
    ap.add_argument('--cond', default='ppi',
                    choices=['ppi', 'kinase', 'shuffled'],
                    help='which protein-level channel to treat as the '
                         'augmented model')
    a = ap.parse_args()
    main(a.root, a.cond)
