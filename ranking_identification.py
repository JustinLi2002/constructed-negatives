#!/usr/bin/env python3
"""What a threshold-sampled benchmark can say about which method is better.

The rest of this repository asks whether a benchmark identifies the effect of
adding a feature.  That is not what benchmarks are for.  They are for ranking
methods, and the ranking is the thing to attack.

The identity transfers verbatim.  For two methods A and B scored against the
same fixed positives and the same negative distribution, AUROC is linear in the
negatives for each, so their difference is too:

    D = AUROC_A - AUROC_B = sum_k w_k D_k ,   D_k = AUROC_A,k - AUROC_B,k

and therefore, with O the strata the benchmark supplies and U those it does not,

    D_deploy - D_O = W_un * (D_bar - D_O) .

An ordering reverses when D_O > 0 > D_deploy, which by the same argument as in
reversal_bound.py requires W_un > D_O / (D_O + 1).  Reading that as a condition
on the *gap* rather than on the weight:

    the ordering is identified only if   D_O >= W_un / (1 - W_un) .

A difference of two AUROCs cannot exceed 1.  So

    **W_un > 0.5  =>  W_un / (1 - W_un) > 1  =>  no gap of any size is
    identified.**

Not "small gaps are unsafe".  No gap at all -- up to and including a comparison
between a perfect ranker and a random one.  Above half the deployment negatives
missing, the unobserved region can overturn any observed ordering, because a
difference of AUROCs can be as low as -1 there and the arithmetic leaves room.

The eight PTM tasks of the first paper sit at W_un = 0.605 to 0.977.

This is not a statistical-power argument and does not go away with more seeds.
The gaps below are means over ten seeds with standard deviations around 0.002;
they are estimated precisely.  They are estimated precisely *for the benchmark's
own negative distribution*, and that distribution is not the one the ordering is
being claimed over.

    python ranking_identification.py /path/to/ptm-audit
"""
import argparse
import itertools
import json
import os

from ptm_guarantee import ALIAS, load_weights


def main(root):
    summ = json.load(open(os.path.join(root, 'results', 'summary.json'),
                         encoding='utf-8'))
    weights = load_weights(os.path.join(root, 'results',
                                        'unobserved_weight.txt'))
    if not weights:
        raise SystemExit('could not parse unobserved_weight.txt')

    auroc, sd = {}, {}
    for r in summ:
        auroc.setdefault(r['ptm'], {})[r['cond']] = r['seed_mean']
        sd.setdefault(r['ptm'], {})[r['cond']] = r.get('seed_sd', float('nan'))

    print('Pairwise method orderings on the first paper\'s eight tasks.\n')
    print('gap        = |AUROC_A - AUROC_B| on the benchmark, mean over 10 seeds')
    print('gap needed = W_un / (1 - W_un), the smallest gap whose ordering is')
    print('             identified; above 1 no gap qualifies, since a difference')
    print('             of AUROCs cannot exceed 1\n')

    print('%-22s %7s %8s %10s %12s %8s'
          % ('task', 'W_un', 'gaps', 'median gap', 'gap needed', 'ident.'))
    print('-' * 72)

    tot = ident = 0
    biggest = (0.0, '', '')
    for name, (w_res, w_dep) in weights.items():
        key = ALIAS.get(name)
        if key not in auroc:
            continue
        conds = sorted(auroc[key])
        gaps = [abs(auroc[key][a] - auroc[key][b])
                for a, b in itertools.combinations(conds, 2)]
        need = w_dep / (1 - w_dep) if w_dep < 1 else float('inf')
        ok = sum(1 for g in gaps if g >= need)
        tot += len(gaps)
        ident += ok
        med = sorted(gaps)[len(gaps) // 2]
        mx = max(gaps)
        if mx > biggest[0]:
            biggest = (mx, name, 'largest observed gap')
        print('%-22s %7.3f %8d %10.4f %12s %5d/%d'
              % (name, w_dep, len(gaps), med,
                 ('%.2f' % need) if need < 100 else '%.0f' % need,
                 ok, len(gaps)))
    print('-' * 72)
    print('orderings whose direction is identified: %d of %d' % (ident, tot))
    print('largest gap anywhere in the table: %.4f (%s)'
          % (biggest[0], biggest[1]))

    print('\nEvery task needs a gap above 1.5 AUROC -- an impossibility -- for '
          'any\nordering to be identified, because every task has more than '
          'half its\ndeployment negatives missing.  The comparison that would '
          'qualify does not\nexist: a perfect ranker against a random one gaps '
          'by 0.5.')

    print('\nSame question in the restricted universe, the most generous '
          'reading:')
    print('%-22s %7s %12s' % ('task', 'W_un res', 'gap needed'))
    print('-' * 44)
    for name, (w_res, w_dep) in weights.items():
        if ALIAS.get(name) not in auroc:
            continue
        need = w_res / (1 - w_res) if w_res < 1 else float('inf')
        print('%-22s %7.3f %12.2f' % (name, w_res, need))
    print('-' * 44)
    print('Phosphorylation Y is the only task where the required gap falls '
          'below 1,\nand it needs 0.32 AUROC -- larger than any gap in the '
          'literature.')


# The published comparison the criterion applies to, from DeepMVP (Wen et al.,
# Nat Methods 2025;22:1857-67).  Its released data is the benchmark the first
# paper audited, and its test-set negatives are "sites of the same type that
# lacked MS/MS evidence for known PTMs from the same proteins" -- the donor
# restriction itself -- so the criterion applies with no translation.
#
# Only numbers that appear as TEXT in that paper are used.  Its Fig. 3 panels
# carry six more tools, but an AUROC read off a figure is not evidence, and a
# comparison that has to be eyeballed has no place in an argument about exact
# identification.
PUBLISHED = [
    ('Phosphorylation S/T', 'DeepMVP 0.95 vs MusiteDeep 0.83', 0.95, 0.83,
     'Fig. 3, independent test set'),
    ('Phosphorylation S/T', 'DeepMVP 0.95 vs MusiteDeep retrained 0.89',
     0.95, 0.89, 'Extended Data Fig. 5a'),
]


def published_case(weights):
    print()
    print('The same criterion on a published head-to-head comparison.')
    print()
    print('%-22s %-42s %6s %8s %9s'
          % ('task', 'comparison', 'gap', 'needed', 'short by'))
    print('-' * 90)
    for task, label, va, vb, where in PUBLISHED:
        w_res, w_dep = weights[task]
        gap = abs(va - vb)
        for uni, w in (('deployment', w_dep), ('restricted', w_res)):
            need = w / (1 - w)
            head = task if uni == 'deployment' else ''
            body = label if uni == 'deployment' else '  (%s universe)' % uni
            print('%-22s %-42s %6.2f %8.2f %8.1fx'
                  % (head, body, gap, need, need / gap))
        print('%-22s   source: %s' % ('', where))
    print('-' * 90)
    print("Neither ordering is identified, in either universe, and neither is")
    print("close.  These are the comparisons the field's flagship tool leads")
    print("with, in the journal that published it.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='path to a checkout of the ptm-audit repo')
    root = ap.parse_args().root
    main(root)
    published_case(load_weights(os.path.join(root, 'results',
                                             'unobserved_weight.txt')))
