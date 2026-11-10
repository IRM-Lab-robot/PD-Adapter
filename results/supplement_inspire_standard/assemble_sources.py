#!/usr/bin/env python3
"""Assemble frozen train40 upstream sources; keep test10 source files untouched."""
import json
from pathlib import Path

import torch

ROOT = Path('/home/user/yangbo_program/new')
OUT = ROOT / 'results/supplement_inspire_standard'
names = json.loads((ROOT / 'eval/inspire_train_test_split.json').read_text())
train = names['train']
test = set(names['test'])
assert len(train) == len(set(train)) == 40 and not (set(train) & test)
gendex = []
for idx in range(8):
    chunk = torch.load(OUT / f'gendex_train_chunk{idx}.pt', map_location='cpu')
    assert [r['object_name'] for r in chunk] == train[5*idx:5*(idx+1)]
    assert all(torch.as_tensor(r['q_dro_order']).shape == (20, 18) for r in chunk)
    gendex.extend(chunk)
dro_original = torch.load(ROOT / 'eval/inspire_train40_zero_shot_b60.pt', map_location='cpu')
assert [r['object_name'] for r in dro_original] == train
assert all(torch.as_tensor(r['q_dro_order']).shape == (60, 18) for r in dro_original)
dro = [{**r, 'q_dro_order': r['q_dro_order'][:20].clone()} for r in dro_original]
torch.save(dro, OUT / 'dro_train40_sources.pt')
torch.save(gendex, OUT / 'gendex_train40_sources.pt')
summary = {'train_objects': train, 'validation_objects': sorted(set(train) - {r['object_name'] for r in torch.load(ROOT / 'eval/original_train30_zero_shot_b20.pt', map_location='cpu')}),
           'source_count_by_generator': {'DRO': 800, 'GenDexGrasp': 800},
           'evaluation_objects_excluded': sorted(test)}
assert len(summary['validation_objects']) == 10
(OUT / 'source_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'source_count_by_generator': summary['source_count_by_generator'], 'validation_objects': summary['validation_objects']}, indent=2))
