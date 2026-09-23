#!/usr/bin/env python3
"""PTM evaluation shift for each fixed model split on empirical negative sets.

A common positive set and one set of positive scores are used in both evaluations.
Score-file disagreement is a numerical sensitivity, not a confidence interval.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prediction_files import index_predictions
from measured_c import ORDER,depth_of,T
from scope_reanalysis import negative_contribution
KEY=['protein','pos','y']


def pair(index,task,fam,ev,split):
    b=pd.read_csv(index[(task,'baseline',ev)][split],sep='\t')
    a=pd.read_csv(index[(task,fam,ev)][split],sep='\t')
    m=b[KEY+['y_pred']].merge(a[KEY+['y_pred']],on=KEY,
                            suffixes=('_base','_aug'),validate='one_to_one')
    if len(m)!=len(a) or len(m)!=len(b):
        raise ValueError('Model pair does not score identical sites')
    return m


def delta(pos,neg):
    if neg.empty or pos.empty:return np.nan
    return float((negative_contribution(pos.y_pred_aug,neg.y_pred_aug)-
                  negative_contribution(pos.y_pred_base,neg.y_pred_base)).mean())


def main(args):
    index=index_predictions(args.root,args.train);rows=[]
    for task in ORDER:
        depths=depth_of(args.data,task)
        for split in sorted(index[(task,args.family,'rebuilt')]):
            b=pair(index,task,args.family,'replica',split)
            r=pair(index,task,args.family,'rebuilt',split)
            pos=r[r.y==1].merge(b.loc[b.y==1,KEY],on=KEY,validate='one_to_one')
            if len(pos)!=int((r.y==1).sum()) or len(pos)!=int((b.y==1).sum()):
                raise ValueError(f'{task}/{split}: positive references differ')
            nr=r[r.y==0].copy();nb=b[b.y==0]
            nr['depth']=nr.protein.map(depths).fillna(0)
            no,nu=nr[nr.depth>=T],nr[nr.depth<T]
            do,du,dr,db=delta(pos,no),delta(pos,nu),delta(pos,nr),delta(pos,nb)
            w=len(nu)/len(nr);mass=w*(du-do);shift=do-db
            common=nb.merge(nr,on=KEY,suffixes=('_b','_r'),validate='one_to_one')
            if len(common)!=len(nb):
                alternate=np.nan;score_drift=np.nan
            else:
                alt=common[KEY].copy()
                for model in ['base','aug']:alt['y_pred_'+model]=common['y_pred_'+model+'_r']
                alternate=delta(pos,alt)
                score_drift=max(float((common['y_pred_'+model+'_b']-common['y_pred_'+model+'_r']).abs().max()) for model in ['base','aug'])
            rows.append(dict(task=task,split=split,family=args.family,W_R=w,
                delta_bench=db,delta_O=do,delta_U=du,delta_R=dr,
                missing_term=mass,within_O_term=shift,
                residual=abs((dr-db)-(mass+shift)),
                matched_bench_negatives=len(common),bench_negatives=len(nb),
                max_score_drift=score_drift,delta_bench_alternate=alternate,
                metric_drift=abs(db-alternate)))
    f=pd.DataFrame(rows)
    f.to_csv(args.out,sep='\t',index=False,float_format='%.12g')
    print(f.to_string(index=False))
    print('maximum decomposition residual',f.residual.max())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path('~/HRP/pdisjoint_runs_v2').expanduser()))
    p.add_argument('--data',default=str(Path('~/HRP/rebuilt').expanduser()))
    p.add_argument('--family',default='esm2')
    p.add_argument('--train',default='replica')
    p.add_argument('--out',default='results_shift_review.tsv')
    a=p.parse_args();main(a)
