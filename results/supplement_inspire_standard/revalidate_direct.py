#!/usr/bin/env python3
"""Fresh-simulate Raw and naive-Phi on exactly the same test10 source targets."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path('/home/user/yangbo_program/new')
OUT = ROOT / 'results/supplement_inspire_standard'
sys.path.insert(0, str(ROOT / 'eval'))
from holdout_isaac_battery import BATTERIES, run_holdout
from native_inspire_cem_search import lsq_project_q
from utils.hand_model import create_hand_model
import torch

SOURCES = {'dro': ROOT / 'eval/original_test10_zero_shot_b20.pt',
           'gendex': ROOT / 'eval/gendex_inspire_mainset_grasps.pt'}
BATCHES = (32,64,128,256,96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', choices=('dro','gendex'), required=True)
    ap.add_argument('--obj', required=True)
    ap.add_argument('--no-padding', action='store_true')
    a = ap.parse_args()
    split = json.loads((ROOT / 'eval/inspire_train_test_split.json').read_text())
    assert a.obj in split['test'][:10]
    source = next(r for r in torch.load(SOURCES[a.gen],map_location='cpu') if r['object_name'] == a.obj)
    q = torch.as_tensor(source['q_dro_order'],dtype=torch.float32)[:20]
    assert q.shape == (20,18)
    hand = create_hand_model('inspire', device=torch.device('cpu'))
    lower,upper = hand.pk_chain.get_joint_limits()
    projected,_ = lsq_project_q(q,hand.get_joint_orders()[6:],lower[6:],upper[6:])
    for method,batch in ((m,b) for m in ('raw','naive_phi') for b in BATCHES):
        ci = BATCHES.index(batch)
        dest = OUT / ('revalidation_unpadded' if a.no_padding else 'revalidation') / a.gen / method / f'{a.obj.replace("+","_")}_cond{ci}.json'
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            continue
        target = q if method == 'raw' else projected
        order=list(range(20))
        if a.no_padding and ci:
            random.Random(20261201+ci).shuffle(order)
        target=target[order]
        batch_q=target if a.no_padding else torch.cat([target,target[-1:].expand(batch-20,-1)],dim=0)
        success,_,distance = run_holdout(a.obj,batch_q, projection_mode='none' if method=='raw' else 'lsq',
            battery=BATTERIES['standard_six_force05_steps6'],use_gpu_pipeline=False)
        keys = sorted(order[i] for i in range(20) if float(distance[i]) <= .02)
        out = {'generator':a.gen,'method':method,'object_name':a.obj,'cond':ci,
            'batch_size':batch,'tail_padded_to_batch_size':not a.no_padding,'shuffle':a.no_padding and ci>0,
            'battery':'standard_six_force05_steps6','threshold':.02,
            'source_file':str(SOURCES[a.gen]),'success_source_indices':keys,'success_sources':len(keys)}
        dest.write_text(json.dumps(out,indent=2,ensure_ascii=False))
        print(f'{a.gen} {a.obj} {method} {ci}: {len(keys)}/20',flush=True)


if __name__ == '__main__':
    main()
