#!/usr/bin/env python3
"""Resume all new-battery Inspire baseline, learned and random searches on test10."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/user/yangbo_program/new')
OUT = ROOT / 'results/supplement_inspire_standard'
SOURCES = {'dro': ROOT/'eval/original_test10_zero_shot_b20.pt',
           'gendex': ROOT/'eval/gendex_inspire_mainset_grasps.pt'}
METHODS = ('g4','pop256','default',*(f'pd_head{i}' for i in range(8)),*(f'random_head{i}' for i in range(8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen',choices=('dro','gendex','all'),default='all')
    args = ap.parse_args()
    objects = json.loads((ROOT/'eval/inspire_train_test_split.json').read_text())['test'][:10]
    env = dict(os.environ,CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',MAX_JOBS='4')
    for gen in (('dro','gendex') if args.gen=='all' else (args.gen,)):
        for arm in METHODS:
            pop, generations = (64,4) if arm=='g4' else ((256,1) if arm=='pop256' else (32,1))
            seed_file = None
            if arm.startswith(('pd_head','random_head')):
                seed_file = OUT / f'{"pd" if arm.startswith("pd_head") else "random"}_{gen}_head{arm.rsplit("head",1)[1]}.pt'
                if not seed_file.exists():
                    raise FileNotFoundError(seed_file)
            for obj in objects:
                target = OUT / 'search' / gen / arm / (obj.replace('+','_')+'.pt')
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    continue
                cmd=[sys.executable,str(ROOT/'eval/generate_inspire_train_bsearch.py'),
                    '--objects',obj,'--zero-file',str(SOURCES[gen]),'--allow-non-train',
                    '--max-grasps','20','--selection-battery','standard_six_force05_steps6',
                    '--population',str(pop),'--elites','8','--generations',str(generations),
                    '--restarts','1','--batch-source-grasps','8','--keep-per-grasp','3',
                    '--keep-per-object','120','--cpu-final-candidates','240',
                    '--search-pipeline','gpu','--reuse-gpu-sim','--log-rejected',
                    '--arm-id',arm,'--seed','10001','--out',str(target)]
                if seed_file:
                    cmd += ['--warm-start-file',str(seed_file)]
                log = target.with_suffix('.log')
                with log.open('w') as handle:
                    proc = subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                if proc.returncode or not target.exists():
                    raise RuntimeError(f'failed {gen} {arm} {obj}; inspect {log}')
                print(f'completed {gen} {arm} {obj}',flush=True)


if __name__=='__main__':
    main()
