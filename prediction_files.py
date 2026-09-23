"""Explicit model-condition indexing shared by the PTM audit scripts."""
import re
from pathlib import Path

PATTERN = (r'^(?P<task>[a-z_0-9]+?)__(?P<train>replica|rebuilt)'
           r'__(?P<cond>baseline|ppi|shuffled)__split(?P<split>\d+)'
           r'(?:__(?P<source>esm|prott5))?'
           r'__on_(?P<eval>replica|rebuilt)\.pred\.tsv$')


def family(condition, source):
    if condition == 'baseline':
        return 'baseline'
    base = {None:'interaction','esm':'esm2','prott5':'prott5'}[source]
    if condition == 'ppi':
        return base
    if condition == 'shuffled':
        return 'shuffled_'+base
    raise ValueError(f'Unknown prediction condition {condition}')


def index_predictions(root, train):
    index={}
    for path in sorted(Path(root).glob('*.pred.tsv')):
        match=re.match(PATTERN,path.name)
        if not match:
            continue
        item=match.groupdict()
        if item['train']!=train:
            continue
        key=(item['task'],family(item['cond'],item['source']),item['eval'])
        group=index.setdefault(key,{})
        if item['split'] in group:
            raise ValueError(f'Duplicate prediction key {key}/{item["split"]}')
        group[item['split']]=str(path)
    return index


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for condition in ['baseline','ppi','shuffled']:
            source='' if condition=='baseline' else '__esm'
            (root/f'task__replica__{condition}__split0{source}__on_rebuilt.pred.tsv').touch()
        index=index_predictions(root,'replica')
        assert '__ppi__' in index[('task','esm2','rebuilt')]['0']
        assert '__shuffled__' in index[('task','shuffled_esm2','rebuilt')]['0']
        (root/'task__replica__baseline__split0__esm__on_rebuilt.pred.tsv').touch()
        try:
            index_predictions(root,'replica')
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate model identity was silently accepted')
    print('prediction_files self-tests: condition isolation and duplicate rejection passed')


if __name__=='__main__':
    self_test()
