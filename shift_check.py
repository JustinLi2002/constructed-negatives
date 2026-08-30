"""How big is the within-O covariate shift term the manuscript assumed away?

    Delta_deploy - Delta_bench = W (Delta_bar_U - Delta_O)   [unobserved mass]
                               + (Delta_O - Delta_bench)     [shift inside O]

Delta_O uses deployment weights on the observed strata; Delta_bench is what the
benchmark actually reports, using its own within-O negative distribution.  The
manuscript treated them as equal.  This measures the gap.
"""
import numpy as np
from cl3_exclusion import config_threshold, load_edges
from ppi_consequence import one_run
from reversal_bound import split_effect

A, names = load_edges('data/string_phys_700.tsv')
deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)

print('%6s %6s | %9s %9s %9s %9s | %10s %10s %10s'
      % ('frac', 'r2', 'W', 'D_bench', 'D_O', 'D_bar_U',
         'unobs term', 'shift term', 'shift/tot'))
print('-' * 104)
for frac in (0.064, 0.2, 0.5):
    tau = config_threshold(deg, frac)
    for r2 in (0.30, 0.90):
        acc = []
        for s in range(5):
            r = one_run(A, deg, tau, s, r2=r2)
            if r is None:
                continue
            w, d_O, d_bar, d_dep = split_effect(r, deg, tau)
            acc.append((w, r['delta_bench'], d_O, d_bar, d_dep))
        if not acc:
            continue
        w, db, dO, dU, dd = (float(np.nanmean([a[i] for a in acc]))
                             for i in range(5))
        unobs = w * (dU - dO)
        shift = dO - db
        tot = dd - db
        print('%6.3f %6.2f | %+9.4f %+9.4f %+9.4f %+9.4f | %+10.4f %+10.4f %9.1f%%'
              % (frac, r2, w, db, dO, dU, unobs, shift,
                 100 * abs(shift) / max(abs(tot), 1e-12)))
