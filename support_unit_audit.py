#!/usr/bin/env python3
"""Recover pair-level support loss from released per-protein eligibility counts.

Each undirected eligible pair is counted at both endpoints. The target here is
all distinct nonedges of the input graph, not its positive-edge degree mass.
No new L3 computation or change to the input networks is needed.
"""
import argparse
import csv
from bisect import bisect_right
from pathlib import Path


def audit(path):
    with open(path, encoding='utf-8') as f:
        rows=list(csv.DictReader(f, delimiter='\t'))
    n=len(rows)
    degrees=[int(r['degree']) for r in rows]
    eligible=[int(r['n_eligible']) for r in rows]
    assert sum(degrees)%2==0 and sum(eligible)%2==0
    assert all(0<=e<=n-1-d for d,e in zip(degrees,eligible))
    candidates=(n*(n-1)-sum(degrees))//2
    admitted=sum(eligible)//2
    ordered=sorted(degrees)
    low,high=1,max(degrees)**2
    target=.064*n*(n-1)/2
    while low<high:
        mid=(low+high)//2
        count=(sum(bisect_right(ordered,mid//d) for d in degrees)-
               sum(d*d<=mid for d in degrees))/2
        if count<target:low=mid+1
        else:high=mid
    return dict(source=str(path),proteins=n,nonedge_pairs=candidates,
                admitted_pairs=admitted,excluded_pair_share=1-admitted/candidates,
                excluded_proteins=sum(e==0 for e in eligible),
                configuration_tau=low,
                proteins_above_tau=sum(d>low for d in degrees))


def self_test():
    # Four-node path: every node has an admitted nonedge partner, but (0,3)
    # is excluded because it completes a length-three path. Marginal coverage
    # therefore cannot establish pair-level positivity.
    adjacent={(0,1),(1,2),(2,3)}
    candidates={(u,v) for u in range(4) for v in range(u+1,4)}-adjacent
    admitted=candidates-{(0,3)}
    assert all(any(u in p for p in admitted) for u in range(4))
    assert len(admitted)==2 and len(candidates)==3
    print('support_unit_audit self-test: passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default='clusterC/results_uniform')
    p.add_argument('--out',default='results_support_unit.tsv')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test()
    else:
        results=[]
        for name in ['string_phys_700','string_phys_900','biogrid_human','hippie_063']:
            r=audit(Path(a.directory)/f'cl3_{name}_per_protein.tsv')
            r={'network':name,**r};results.append(r)
            print(name,'pair exclusion',f"{100*r['excluded_pair_share']:.2f}%",
                  'excluded proteins',r['excluded_proteins'])
        with open(a.out,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]),delimiter='\t')
            w.writeheader();w.writerows(results)
