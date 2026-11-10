#!/usr/bin/env python3
"""Resume all standard-battery searches' five-condition CPU-state verification."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path('/home/user/yangbo_program/new')
OUT=ROOT/'results/supplement_inspire_standard'
OBJECTS=json.loads((ROOT/'eval/inspire_train_test_split.json').read_text())['test'][:10]
METHODS=('g4','pop256','pd9','random9')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gen',choices=('dro','gendex','all'),default='all')
    ap.add_argument('--no-padding',action='store_true')
    args=ap.parse_args()
    label_dir='revalidation_unpadded' if args.no_padding else 'revalidation'
    env=dict(os.environ,CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',MAX_JOBS='4')
    for gen in (('dro','gendex') if args.gen=='all' else (args.gen,)):
        for method in METHODS:
            for obj in OBJECTS:
                for ci in range(5):
                    dest=OUT/label_dir/gen/method/f'{obj.replace("+","_")}_cond{ci}.json'
                    if dest.exists():
                        continue
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    cmd=[sys.executable,str(OUT/'revalidate_search.py'),
                        '--gen',gen,'--method',method,'--obj',obj,'--cond',str(ci)]
                    if args.no_padding:
                        cmd.append('--no-padding')
                    log=dest.with_suffix('.log')
                    with log.open('w') as handle:
                        proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                    if proc.returncode or not dest.exists():
                        raise RuntimeError(f'failed {gen}/{method}/{obj}/{ci}; inspect {log}')
                    print(f'completed {gen} {method} {obj} cond{ci}',flush=True)


if __name__=='__main__':
    main()
