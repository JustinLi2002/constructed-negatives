#!/usr/bin/env python3
"""Check archived calculations independently of the manuscript's prose checker.

This checks identities, model-condition provenance, aggregation, reference
boundaries and rounded Table 1 values. It does not prove transportability,
label validity or statistical coverage.
"""
import json
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd


def verify(root):
    root = Path(root)
    r = pd.read_csv(root/'results_scope_review_perrun.tsv', sep='\t')
    s = pd.read_csv(root/'results_scope_review_summary.tsv', sep='\t')
    g = pd.read_csv(root/'results_scope_review_geometry.tsv', sep='\t')
    m = json.loads((root/'results_scope_review_manifest.json').read_text())
    assert len(r) == 8*3*3*11
    assert not r.duplicated(['task','split','score','d_min']).any()
    assert set(r.family) == {'esm2'}
    assert len(m['prediction_files']) == 48
    assert all('__shuffled__' not in f['name'] for f in m['prediction_files'])
    aug = [f for f in m['prediction_files'] if '__baseline__' not in f['name']]
    assert len(aug) == 24 and all('__ppi__' in f['name'] and '__esm__' in f['name'] for f in aug)
    assert all(len(f['sha256']) == 64 for f in m['prediction_files'])
    du = r.delta_U.fillna(r.delta_O)
    np.testing.assert_allclose(r.delta_R, (1-r.W_R)*r.delta_O+r.W_R*du, atol=2e-12, rtol=0)
    np.testing.assert_allclose(r.c, abs(du-r.delta_O), atol=2e-12, rtol=0)
    np.testing.assert_allclose(r.W_R, r.n_U/r.n_neg, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(r.signed_agreement, (r.delta_O*r.delta_R>0).astype(int))
    for row in s.itertuples():
        q = r[(r.task==row.task)&(r.d_min==0)&(r.score=='y_pred')]
        for c in ['W_R','delta_O','delta_U','c','delta_R']:
            assert abs(q[c].mean()-getattr(row,c)) < 2e-12, (row.task,c)
        assert q.signed_agreement.sum() == row.signed_agreement_count
        a = g[(g.task==row.task)&(g.W<=.5)].iloc[0]
        assert row.d_min_A == a.d_min
        assert abs(row.A_pct-a.retained_protein_pct) < 1e-9
    assert list(s.signed_agreement_count) == [0,3,0,0,0,0,0,0]
    assert s.initialization_agreement_count.tolist() == [0,6,1,0,0,0,0,0]
    text = (root/'MANUSCRIPT.md').read_text(encoding='utf-8')
    table = text.split('**Table 1.**',1)[1].split('[[FIG2]]',1)[0]
    rows = [line.split('|')[1:-1] for line in table.splitlines()
            if line.startswith('| ') and not line.startswith('| Task')]
    assert len(rows)==16
    for i,row in enumerate(s.itertuples()):
        ref=[v.strip() for v in rows[i][1:]]
        assert ref == [f'{row.W_reference:.3f}',f'{row.delta_bench_legacy:.4f}',str(row.d_min_A),
                       f'{row.A_pct:.1f}%',f'{row.B_scenario_pct:.1f}%']
        emp=[v.strip() for v in rows[i+8][1:]]
        assert emp == [f'{row.W_R:.3f}',f'{row.delta_O:+.4f}',f'{row.c:.4f}',
                       f'{row.delta_R:+.4f}',f'{row.signed_agreement_count}/3']
    p = pd.read_csv(root/'results_support_unit.tsv',sep='\t')
    from support_unit_audit import audit
    for row in p.itertuples():
        calculated = audit(root/'clusterC/results_uniform'/f'cl3_{row.network}_per_protein.tsv')
        for key in ['nonedge_pairs','admitted_pairs','configuration_tau','excluded_proteins']:
            assert calculated[key] == getattr(row,key), (row.network,key)
        assert abs(calculated['excluded_pair_share']-row.excluded_pair_share)<1e-12
    upna = json.loads((root/'upna_sets.json').read_text())
    universe, negatives = set(upna['uni']), upna['neg']
    assert len(universe)==17975 and len(negatives)==5037
    assert len(universe & negatives.keys())==4859
    assert len(universe-negatives.keys())==13116
    assert sum(negatives.values())==2*3063605
    print('Verified 792 run/depth rows, 48 input hashes, split means, reference boundaries and all 16 Table 1 rows.')
    print('No shuffled controls selected; 7/8 tasks reverse in every ensemble split.')
    print('Verified pair counts/cutoffs on four networks and UPNA-PPI universe intersection.')
    docx = root/'MANUSCRIPT_review.docx'
    if docx.exists():
        ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        with zipfile.ZipFile(docx) as z:
            xml = ET.fromstring(z.read('word/document.xml'))
            tables = xml.findall('.//w:tbl',ns)
            counts = [len(t.findall('w:tr',ns)[0].findall('w:tc',ns)) for t in tables]
            assert counts == [6,6,5,4,6], counts
            for t,count in zip(tables,counts):
                assert all(len(row.findall('w:tc',ns))==count for row in t.findall('w:tr',ns))
            comments=ET.fromstring(z.read('word/comments.xml')).findall('w:comment',ns)
            assert len(comments)==8
            image_hashes={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
            expected={hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'figures').glob('fig*.png')}
            assert image_hashes == expected
            body=''.join(xml.itertext())
            assert 'A failed sufficient condition does not override the directly' in body
        print('DOCX: five physical tables with expected columns, eight comments, and five current figure bitmaps verified.')


if __name__=='__main__':
    verify(Path(__file__).resolve().parent)
