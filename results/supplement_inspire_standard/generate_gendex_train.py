#!/usr/bin/env python3
"""Resume frozen GenDex Inspire train40 source generation in 5-object chunks."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/user/yangbo_program/new')
OUT = ROOT / 'results/supplement_inspire_standard'
GENDEX = Path('/home/user/yangbo_program/GenDexGrasp')


def main():
    objs = json.loads((ROOT / 'eval/inspire_train_test_split.json').read_text())['train']
    assert len(objs) == len(set(objs)) == 40
    for idx in range(8):
        part = objs[idx*5:(idx+1)*5]
        dest = OUT / f'gendex_train_chunk{idx}.pt'
        if dest.exists():
            print(f'already exists {dest}', flush=True)
            continue
        cmd = [sys.executable, 'run_inspire_mainset.py', '--objects', *part,
               '--num-per-object', '20', '--seed', '42', '--out', str(dest)]
        log = OUT / f'gendex_train_chunk{idx}.log'
        with log.open('w') as handle:
            proc = subprocess.run(cmd, cwd=GENDEX, stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode or not dest.exists():
            raise RuntimeError(f'chunk {idx} failed, inspect {log}')
        print(f'finished {idx} {part}', flush=True)


if __name__ == '__main__':
    main()
