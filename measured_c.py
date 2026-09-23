#!/usr/bin/env python3
"""Measure c separately for each fixed model pair on its empirical negatives.

Use scope_reanalysis.py for the complete reference coverage and per-depth scan.
Shuffled controls cannot overwrite augmented predictions. No reference W is
multiplied by empirical regional effects.
"""
import argparse
from pathlib import Path
import pandas as pd
from prediction_files import PATTERN, family, index_predictions
from deployment_scope import MERGED

PROTEIN,SITE,LABEL,SCORE='protein','pos','y','y_pred'
ORDER=['phosphorylation_st','phosphorylation_y','acetylation_k','methylation_k',
       'methylation_r','sumoylation_k','ubiquitination_k','glycosylation_n']
T=10


def depth_of(data_dir,task):
    counts={}
    for source in MERGED[task]:
        df=pd.read_csv(Path(data_dir)/f'{source}_all.tsv',sep='\t',usecols=[PROTEIN,SITE,LABEL])
        for protein,group in df[df.y==1].groupby(PROTEIN):
            counts[protein]=counts.get(protein,0)+group.pos.nunique()
    return pd.Series(counts,dtype=int)


def main(root,data,fam,train,ev):
    from scope_reanalysis import negative_contribution,effect_at_scope
    index=index_predictions(root,train)
    print('Empirical reconstruction only; each split is a fixed model pair.')
    print('family=%s train=%s eval=%s threshold=%d'%(fam,train,ev,T))
    print('task\tsplit\tW_R\tdelta_O\tdelta_U\tc\tdelta_R\tresidual')
    for task in ORDER:
        base=index.get((task,'baseline',ev),{})
        aug=index.get((task,fam,ev),{})
        if not base or set(base)!=set(aug):
            raise ValueError(f'Missing or unmatched splits for {task}/{fam}')
        depth=depth_of(data,task)
        for split in sorted(base):
            b=pd.read_csv(base[split],sep='\t');a=pd.read_csv(aug[split],sep='\t')
            m=b.merge(a,on=[PROTEIN,SITE,LABEL],suffixes=('_base','_aug'),validate='one_to_one')
            if len(m)!=len(a) or len(m)!=len(b):
                raise ValueError('Models scored different sites')
            pos=m.y==1;neg=~pos
            g=negative_contribution(m.loc[pos,'y_pred_aug'],m.loc[neg,'y_pred_aug'])-negative_contribution(m.loc[pos,'y_pred_base'],m.loc[neg,'y_pred_base'])
            r=effect_at_scope(g,m.loc[neg,PROTEIN].map(depth).fillna(0).to_numpy(),0,T)
            print(task,split,*[f'{r[k]:.12g}' for k in ['W_R','delta_O','delta_U','c','delta_R','residual']],sep='\t')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path('~/HRP/pdisjoint_runs_v2').expanduser()))
    p.add_argument('--data',default=str(Path('~/HRP/rebuilt').expanduser()))
    p.add_argument('--family',default='esm2')
    p.add_argument('--train',default='replica')
    p.add_argument('--eval',default='rebuilt')
    a=p.parse_args();main(a.root,a.data,a.family,a.train,a.eval)
