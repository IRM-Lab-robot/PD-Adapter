#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/user/yangbo_program/new')
OUT = ROOT / 'results/supplement_inspire_standard'
objects = json.loads((ROOT / 'eval/inspire_train_test_split.json').read_text())['test'][:10]
ap=argparse.ArgumentParser()
ap.add_argument('--no-padding',action='store_true')
a=ap.parse_args()
label_dir='revalidation_unpadded' if a.no_padding else 'revalidation'
env = dict(os.environ, CC='/usr/bin/gcc-9', CXX='/usr/bin/g++-9', MAX_JOBS='4')
for gen in ('dro','gendex'):
    for obj in objects:
        dest = OUT / label_dir / gen / 'naive_phi' / f'{obj.replace("+","_")}_cond4.json'
        if dest.exists():
            continue
        log = OUT / f'direct_{label_dir}_{gen}_{obj.replace("+","_")}.log'
        cmd=[sys.executable,str(OUT/'revalidate_direct.py'),'--gen',gen,'--obj',obj]
        if a.no_padding:
            cmd.append('--no-padding')
        with log.open('w') as handle:
            proc = subprocess.run(cmd,
                                  cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        if proc.returncode or not dest.exists():
            raise RuntimeError(f'failed {gen} {obj}; inspect {log}')
        print(f'completed {gen} {obj}',flush=True)
