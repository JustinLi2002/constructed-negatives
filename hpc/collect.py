#!/usr/bin/env python3
"""Merge the array tasks' TSVs into one table and print the plane.

    python hpc/collect.py results/ > results/plane.txt

Reports the prefilter-severity x feature-fidelity plane for each network: what
the benchmark reads, what deployment reads, how much of the deployment mass the
benchmark supplies nothing for, and how often the two disagree in sign.  A
prefilter fraction of 1.0 keeps every non-edge and so recovers random sampling,
which is the control column -- if the plane shows nothing at 1.0 and something
at 0.064, the something is attributable to the rule.
"""
import glob
import os
import sys
from collections import defaultdict


def main(root):
    rows = defaultdict(list)
    files = sorted(glob.glob(os.path.join(root, 'ppi_*.tsv')))
    if not files:
        sys.exit('no ppi_*.tsv under %s' % root)

    for path in files:
        net = os.path.basename(path)[4:].rsplit('_f', 1)[0]
        with open(path, encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            for line in fh:
                v = line.rstrip('\n').split('\t')
                if len(v) != len(head):
                    continue
                rows[net].append(dict(zip(head, v)))

    for net in sorted(rows):
        rs = rows[net]
        fracs = sorted({float(r['prefilter_frac']) for r in rs})
        r2s = sorted({float(r['r2']) for r in rs})
        print('\n=== %s   (%d cells)' % (net, len(rs)))
        print('deployment minus benchmark: how far the benchmark reading is '
              'from the truth\n')
        print('%8s | %s' % ('frac', ''.join('%12s' % ('r2=%.2f' % r)
                                            for r in r2s)))
        print('-' * (10 + 12 * len(r2s)))
        for f in fracs:
            cells = []
            for r2 in r2s:
                m = [r for r in rs
                     if float(r['prefilter_frac']) == f and float(r['r2']) == r2]
                if not m:
                    cells.append('%12s' % '--')
                    continue
                d = float(m[0]['delta_deploy']) - float(m[0]['delta_bench'])
                flip = int(m[0]['n_flip'])
                cells.append('%11.4f%s' % (d, '*' if flip else ' '))
            tag = '%.3f' % f + (' (ctrl)' if f >= 1.0 else '')
            print('%8s | %s' % (tag, ''.join(cells)))
        print('\n* = at least one replicate where the benchmark says help and '
              'deployment says harm')

        wun = {}
        for f in fracs:
            m = [r for r in rs if float(r['prefilter_frac']) == f]
            if m:
                wun[f] = sum(float(r['w_unobs']) for r in m) / len(m)
        print('\nweight the benchmark supplies nothing for, by severity:')
        for f in fracs:
            if f in wun:
                print('  frac %.3f  tau %6s   w_unobs %.3f'
                      % (f, [r['tau'] for r in rs
                             if float(r['prefilter_frac']) == f][0], wun[f]))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results')
