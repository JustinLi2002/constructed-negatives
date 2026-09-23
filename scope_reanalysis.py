#!/usr/bin/env python3
"""Audit scope without transferring effects between populations or model pairs.

Geometry refers to candidate residues in the reference FASTA. Measured effects
refer ONLY to the supplied prediction files. Each model split and initialization
is evaluated separately; positives stay fixed when the negative scope changes.
No reference-proteome W is multiplied by a prediction-sample c.

python scope_reanalysis.py --base /path/to/HRP --out results_scope_review
python scope_reanalysis.py --self-test
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deployment_scope import TARGETS, MERGED, read_fasta

TASKS = ['phosphorylation_st', 'phosphorylation_y', 'ubiquitination_k',
         'sumoylation_k', 'acetylation_k', 'methylation_k', 'methylation_r',
         'glycosylation_n']
KEY = ['protein', 'pos', 'y']
SCORES = ['y_pred', 'init_0', 'init_1']
PUBLISHED = dict(zip(TASKS, [.0165, .0532, .0497, .0667, .0578, .0378, .0534, .0100]))


def negative_contribution(positive_scores, negative_scores):
    """AUROC contribution of each negative, including half-credit for ties."""
    p = np.sort(np.asarray(positive_scores, float))
    n = np.asarray(negative_scores, float)
    if not len(p) or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError('Positive scores must be nonempty and all scores finite')
    left, right = np.searchsorted(p, n, side='left'), np.searchsorted(p, n, side='right')
    return (len(p) - right + .5 * (right - left)) / len(p)


def effect_at_scope(g, depth, minimum, threshold):
    """Recompute the signed effect AND c for each retained negative population."""
    keep = depth >= minimum
    obs, unseen = keep & (depth >= threshold), keep & (depth < threshold)
    n, no, nu = int(keep.sum()), int(obs.sum()), int(unseen.sum())
    if not n or not no:
        return None
    do = float(g[obs].mean())
    du = float(g[unseen].mean()) if nu else do
    w = nu / n
    c = abs(du - do)
    dr = float(g[keep].mean())
    residual = abs(dr - ((1-w)*do + w*du))
    return dict(n_neg=n, n_O=no, n_U=nu, W_R=w, delta_O=do,
                delta_U=du if nu else None, c=c, delta_R=dr,
                residual=residual, signed_agreement=int(do*dr > 0),
                radius_certificate=int(abs(do) > w*c),
                radius_margin=abs(do)-w*c)


def load_annotations(base):
    raw, positive = {}, {}
    for task in TASKS:
        frame = pd.read_csv(base/'rebuilt'/f'{task}_all.tsv', sep='\t', usecols=KEY)
        if frame.duplicated(KEY).any():
            raise ValueError(f'{task}: repeated sites in reconstruction')
        raw[task] = frame
        positive[task] = frame.loc[frame.y == 1].groupby('protein').pos.agg(set).to_dict()
    depths = {}
    for task in TASKS:
        count = {}
        # The donor rule merges phosphorylation ST/Y and methylation K/R.
        for merged_task in MERGED[task]:
            for protein, positions in positive[merged_task].items():
                count[protein] = count.get(protein, 0) + len(positions)
        depths[task] = count
    return raw, positive, depths


def geometry(fasta, positive, depth, task, threshold):
    rows = []
    for protein, sequence in fasta.items():
        nc = sum(sequence.count(aa) for aa in TARGETS[task])
        if not nc:
            continue
        nn = nc - len(positive.get(protein, ()))
        if nn < 0:
            raise ValueError(f'{task}/{protein}: annotated count exceeds candidate count')
        rows.append((protein, depth.get(protein, 0), nn))
    df = pd.DataFrame(rows, columns=['protein', 'depth', 'negative_candidates'])
    output = []
    for d in range(threshold+1):
        keep = df.depth >= d
        mass = df.loc[keep, 'negative_candidates'].sum()
        output.append(dict(task=task, d_min=d,
            n_reference_proteins=len(df), n_retained_proteins=int(keep.sum()),
            retained_protein_pct=100*keep.mean(),
            negative_mass=int(mass),
            W=float(df.loc[keep & (df.depth < threshold), 'negative_candidates'].sum()/mass)))
    return df, output


def paired_predictions(root, task, family, split, train, evaluation):
    suffix = {'esm2': '__esm', 'prott5': '__prott5', 'interaction': ''}[family]
    bp = root/f'{task}__{train}__baseline__split{split}__on_{evaluation}.pred.tsv'
    ap = root/f'{task}__{train}__ppi__split{split}{suffix}__on_{evaluation}.pred.tsv'
    if not bp.exists() or not ap.exists():
        return None, []
    b, a = pd.read_csv(bp, sep='\t'), pd.read_csv(ap, sep='\t')
    if b.duplicated(KEY).any() or a.duplicated(KEY).any():
        raise ValueError(f'{task}/{split}: repeated site within a model split')
    m = b[KEY+SCORES].merge(a[KEY+SCORES], on=KEY, suffixes=('_base','_aug'),
                           validate='one_to_one')
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(f'{task}/{split}: model pair does not score identical sites')
    return m, [bp, ap]


def main(args):
    base = Path(args.base).expanduser()
    root = Path(args.root).expanduser() if args.root else base/'pdisjoint_runs_v2'
    fasta_path = base/'deepmvp/DeepMVP/data/swiss_prot_human_20190214.fasta'
    fasta = read_fasta(str(fasta_path))
    raw, positive, depths = load_annotations(base)
    all_geometry, all_effects, coverage, used = [], [], [], set()
    for task in TASKS:
        target, grow = geometry(fasta, positive[task], depths[task], task, args.threshold)
        all_geometry.extend(grow)
        for split in range(args.splits):
            m, paths = paired_predictions(root,task,args.family,split,args.train,'rebuilt')
            if m is None:
                raise FileNotFoundError(f'Missing prediction pair for {task}/split{split}')
            used.update(paths)
            dep = m.protein.map(depths[task]).fillna(0).to_numpy(int)
            neg = m.y.to_numpy() == 0
            pos = ~neg
            sampled_proteins = set(m.protein)
            coverage.append(dict(task=task, split=split,
                n_rebuilt_proteins=int(raw[task].protein.nunique()),
                n_prediction_proteins=len(sampled_proteins),
                n_negative_proteins=int(m.loc[neg,'protein'].nunique()),
                n_neg_depth_zero=int((neg & (dep==0)).sum()),
                n_positive=int(pos.sum()), n_negative=int(neg.sum()),
                n_reference_proteins=len(target),
                covered_reference_negative_mass=float(target.loc[target.protein.isin(sampled_proteins),
                   'negative_candidates'].sum()/target.negative_candidates.sum())))
            for score in SCORES:
                gb = negative_contribution(m.loc[pos,score+'_base'],m.loc[neg,score+'_base'])
                ga = negative_contribution(m.loc[pos,score+'_aug'],m.loc[neg,score+'_aug'])
                g = ga-gb
                for minimum in range(args.threshold+1):
                    vals = effect_at_scope(g, dep[neg], minimum, args.threshold)
                    if vals is not None:
                        all_effects.append(dict(task=task,family=args.family,train=args.train,
                            evaluation='rebuilt',split=split,score=score,d_min=minimum,
                            n_positive=int(pos.sum()),**vals))
        print(task, 'complete', flush=True)
    gf, ef, cf = pd.DataFrame(all_geometry),pd.DataFrame(all_effects),pd.DataFrame(coverage)
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    gf.to_csv(str(prefix)+'_geometry.tsv',sep='\t',index=False,float_format='%.12g')
    ef.to_csv(str(prefix)+'_perrun.tsv',sep='\t',index=False,float_format='%.12g')
    cf.to_csv(str(prefix)+'_coverage.tsv',sep='\t',index=False,float_format='%.12g')
    summaries=[]
    for task in TASKS:
        geom=gf[gf.task==task]
        whole=geom.iloc[0]
        feasible=geom[geom.W <= .5].iloc[0]
        hypothetical=PUBLISHED[task]/(1+PUBLISHED[task])
        scenario=geom[geom.W <= hypothetical].iloc[0]
        e=ef[(ef.task==task)&(ef.d_min==0)&(ef.score=='y_pred')]
        initial=ef[(ef.task==task)&(ef.d_min==0)&(ef.score!='y_pred')]
        row=dict(task=task,W_reference=float(whole.W),delta_bench_legacy=PUBLISHED[task],
            d_min_A=int(feasible.d_min),A_pct=float(feasible.retained_protein_pct),
            B_scenario_pct=float(scenario.retained_protein_pct),
            n_splits=len(e),signed_agreement_count=int(e.signed_agreement.sum()),
            initialization_agreement_count=int(initial.signed_agreement.sum()),
            n_initialization_runs=len(initial))
        for col in ['W_R','delta_O','delta_U','c','delta_R','radius_margin']:
            row[col]=float(e[col].mean())
            row[col+'_min']=float(e[col].min())
            row[col+'_max']=float(e[col].max())
        # Report the first empirical scope with sign agreement across runs.
        # This describes the tested negatives, not a guaranteed proteome scope.
        first=None
        for d in range(args.threshold+1):
            rs=ef[(ef.task==task)&(ef.d_min==d)&(ef.score!='y_pred')]
            if len(rs)==args.splits*2 and rs.signed_agreement.all():
                first=d;break
        row['first_empirical_sign_agreement_depth']=first
        summaries.append(row)
    sf=pd.DataFrame(summaries)
    sf.to_csv(str(prefix)+'_summary.tsv',sep='\t',index=False,float_format='%.12g')
    manifest=dict(threshold=args.threshold,family=args.family,train=args.train,
                  estimand='Per-split empirical AUROC difference, fixed positives; no full-proteome extrapolation',
                  aggregation='Unweighted means of split-specific ensemble statistics; mean(W*c) is not mean(W)*mean(c)',
                  max_identity_residual=float(ef.residual.max()),
                  fasta_sha256=hashlib.sha256(fasta_path.read_bytes()).hexdigest(),
                  prediction_files=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                    for p in sorted(used)])
    Path(str(prefix)+'_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(sf[['task','W_reference','A_pct','W_R','delta_O','delta_U','c','delta_R',
              'signed_agreement_count','first_empirical_sign_agreement_depth']].to_string(index=False))
    print('max identity residual',manifest['max_identity_residual'])


def self_test():
    # W < 1/2 permits some guarantee, but does not certify a small actual gap.
    assert .6*.01-.4 < 0 < .6*.01+.4
    # Exact pairwise AUC including ties, tested independently of sorting formula.
    p,n=np.array([0.,1.,2.]),np.array([0.,.5,2.,3.])
    direct=np.array([np.mean((p>x)+.5*(p==x)) for x in n])
    np.testing.assert_allclose(negative_contribution(p,n),direct)
    # Same-sign unobserved effect can strengthen an ordering while failing a
    # symmetric sufficient certificate. Failure is NOT an observed reversal.
    r=effect_at_scope(np.array([-.08,-.22,-.22]),np.array([10,1,1]),0,10)
    assert r['signed_agreement'] and not r['radius_certificate']
    # A full-U average is not a uniform bound for subsets. Recompute c.
    g=np.array([.1,.9,-.7]); dep=np.array([10,1,2])
    r0=effect_at_scope(g,dep,0,10);r2=effect_at_scope(g,dep,2,10)
    assert r0['c']<1e-12 and r2['c']>.79
    assert r0['signed_agreement'] and not r2['signed_agreement']
    assert max(r0['residual'],r2['residual'])<1e-12
    print('scope_reanalysis self-tests: 4 passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='~/HRP')
    parser.add_argument('--root')
    parser.add_argument('--out',default='results_scope_review')
    parser.add_argument('--family',choices=['esm2','interaction','prott5'],default='esm2')
    parser.add_argument('--train',choices=['replica','rebuilt'],default='replica')
    parser.add_argument('--splits',type=int,default=3)
    parser.add_argument('--threshold',type=int,default=10)
    parser.add_argument('--self-test',action='store_true')
    a=parser.parse_args()
    self_test() if a.self_test else main(a)
