#!/usr/bin/env python3
"""Revalidate saved Inspire pools under Ability/SVH six-force battery."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT=Path('/home/user/yangbo_program/new')
OUT=ROOT/'results/supplement_inspire_standard'
sys.path.insert(0,str(ROOT/'eval'))
# validator imports IsaacGym before torch
import generate_inspire_train_bsearch as search
import torch

BATCHES=(32,64,128,256,96)


def arms(method):
    if method in ('g4','pop256'):
        return (method,)
    prefix='pd' if method=='pd9' else 'random'
    return ('default',*(f'{prefix}_head{i}' for i in range(8)))


def revalidate(gen,method,obj,ci,no_padding=False):
    label_dir='revalidation_unpadded' if no_padding else 'revalidation'
    dst=OUT/label_dir/gen/method/f'{obj.replace("+","_")}_cond{ci}.json'
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        return
    candidates=[]
    for arm in arms(method):
        source=OUT/'search'/gen/arm/f'{obj.replace("+","_")}.pt'
        if not source.exists():
            raise FileNotFoundError(source)
        data=torch.load(source,map_location='cpu')
        assert data['total_source_grasps']==20 and data['objects']==[obj]
        for label,group in [('passing',data['rows'][0]['passing_candidates']),('rejected',data['rows'][0].get('cpu_final_rejected_candidates',[]))]:
            for cand in group:
                candidates.append((int(cand['source_grasp_index']),torch.as_tensor(cand['q'],dtype=torch.float32),arm,label))
    batch=BATCHES[ci]
    order=list(range(len(candidates)))
    if ci:
        random.Random(20261201+ci).shuffle(order)
    keys=set()
    for start in range(0,len(order),batch):
        part=order[start:start+batch]
        q=torch.stack([candidates[i][1] for i in part])
        if len(part)<batch and not no_padding:
            q=torch.cat([q,q[-1:].expand(batch-len(part),-1)],dim=0)
        distances,success=search.score_fresh(obj,q,'standard_six_force05_steps6',use_gpu_pipeline=False)
        for j,idx in enumerate(part):
            if float(distances[j])<=.02:
                keys.add(candidates[idx][0])
    row={'generator':gen,'method':method,'object_name':obj,'cond':ci,
        'batch_size':batch,'shuffle':ci>0,'shuffle_seed':20261201+ci,
        'tail_padded_to_batch_size':not no_padding,'battery':'standard_six_force05_steps6','threshold':.02,
        'cpu_candidates':len(candidates),'success_source_indices':sorted(keys),'success_sources':len(keys)}
    dst.write_text(json.dumps(row,indent=2))
    print(f'{gen} {method} {obj} cond{ci}: {len(keys)} successes / {len(candidates)} candidates',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--gen',choices=('dro','gendex'),required=True)
    ap.add_argument('--method',choices=('g4','pop256','pd9','random9'),required=True)
    ap.add_argument('--obj',required=True)
    ap.add_argument('--cond',type=int,choices=range(5),required=True)
    ap.add_argument('--no-padding',action='store_true')
    a=ap.parse_args()
    assert a.obj in json.loads((ROOT/'eval/inspire_train_test_split.json').read_text())['test'][:10]
    revalidate(a.gen,a.method,a.obj,a.cond,no_padding=a.no_padding)
