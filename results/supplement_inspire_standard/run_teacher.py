#!/usr/bin/env python3
"""Generate mixed Inspire teacher/validation pools using the six-direction protocol."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/user/yangbo_program/new')
OUT = ROOT / 'results/supplement_inspire_standard'


def main():
    split = json.loads((ROOT / 'eval/inspire_train_test_split.json').read_text())
    objects = split['train']
    assert len(objects) == 40 and not set(objects) & set(split['test'])
    env = dict(os.environ, CC='/usr/bin/gcc-9', CXX='/usr/bin/g++-9', MAX_JOBS='4')
    for gen in ('dro', 'gendex'):
        source = OUT / f'{gen}_train40_sources.pt'
        assert source.exists()
        for obj in objects:
            target = OUT / 'teacher' / gen / (obj.replace('+', '_') + '.pt')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                print(f'existing {gen} {obj}', flush=True)
                continue
            cmd = [sys.executable, str(ROOT / 'eval/generate_inspire_train_bsearch.py'),
                   '--zero-file', str(source), '--objects', obj, '--max-grasps', '20',
                   '--selection-battery', 'standard_six_force05_steps6',
                   '--population', '64', '--elites', '8', '--generations', '4', '--restarts', '1',
                   '--batch-source-grasps', '8', '--keep-per-grasp', '3', '--keep-per-object', '120',
                   '--search-pipeline', 'gpu', '--reuse-gpu-sim', '--cpu-final-candidates', '240',
                   '--log-rejected', '--seed', '10001', '--arm-id', f'teacher_{gen}', '--out', str(target)]
            log = target.with_suffix('.log')
            with log.open('w') as handle:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if proc.returncode != 0 or not target.exists():
                raise RuntimeError(f'{gen} {obj} failed; inspect {log}')
            print(f'completed {gen} {obj}', flush=True)


if __name__ == '__main__':
    main()
