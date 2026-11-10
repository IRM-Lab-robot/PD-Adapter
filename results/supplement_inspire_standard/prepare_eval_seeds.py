#!/usr/bin/env python3
"""Freeze standard-battery Inspire checkpoint and match random diversity on val10 only."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/home/user/yangbo_program/new')
OUT = ROOT / 'results/supplement_inspire_standard'
sys.path.insert(0, str(ROOT / 'src/eval'))
import generate_pi_phi_warmstart_step12 as warm

MODEL = OUT / 'train/best_pi_phi_v5_multihead.pt'
SOURCES = {'dro': ROOT / 'eval/original_test10_zero_shot_b20.pt',
           'gendex': ROOT / 'eval/gendex_inspire_mainset_grasps.pt'}
VALIDATION = {'dro': OUT / 'dro_train40_sources.pt',
              'gendex': OUT / 'gendex_train40_sources.pt'}
RNG_SEED = 20262041


def pair_l2(p):
    return torch.pdist(p, p=2).mean().item()


def proposals(rows, names, model, finger_names, lo, hi, validation=False):
    output = []
    with torch.no_grad():
        for row in rows:
            obj = row['object_name']
            if obj not in names:
                continue
            raw = torch.as_tensor(row['q_dro_order'], dtype=torch.float32)[:20]
            assert raw.shape == (20,18), obj
            q_seed, u_seed = warm.lsq_project_q_local(raw, finger_names, lo, hi)
            pc = warm.load_object_points(obj).unsqueeze(0)
            for sid in range(20):
                w = q_seed[sid,:6].reshape(1,6)
                pred = model(pc, q_seed[sid].reshape(1,18), w)
                learned = torch.cat([pred['pred_w'][0], pred['pred_u'][0]], dim=1).cpu().float()
                source = torch.cat([w[0],u_seed[sid]]).float()
                output.append((obj, sid, learned, source))
    return output


def main():
    split = json.loads((ROOT / 'eval/inspire_train_test_split.json').read_text())
    train_names = {r['object_name'] for r in torch.load(ROOT / 'eval/original_train30_zero_shot_b20.pt', map_location='cpu')}
    val_names = set(split['train']) - train_names
    test_names = set(split['test'][:10])
    assert len(val_names) == 10 and len(test_names) == 10 and not ((train_names | val_names) & test_names)
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    warm.MODEL_PATH = MODEL
    model, ckpt = warm.load_model()
    assert ckpt['K'] == 8
    hand = warm.create_hand_model('inspire',device=torch.device('cpu'))
    finger_names = hand.get_joint_orders()[6:]
    lower, upper = hand.pk_chain.get_joint_limits()
    lower, upper = lower[6:], upper[6:]
    drive_lo = torch.tensor(ckpt['active_lower']).reshape(1,6)
    drive_hi = torch.tensor(ckpt['active_upper']).reshape(1,6)
    val = []
    for gen, src in VALIDATION.items():
        rows = torch.load(src,map_location='cpu')
        val.extend(proposals(rows, val_names, model, finger_names, lower, upper))
    assert len(val) == 400
    target = sum(pair_l2(pred) for _,_,pred,_ in val)/len(val)
    generator = torch.Generator().manual_seed(RNG_SEED)
    noise = torch.randn((len(val),8,12),generator=generator)
    low, high = 0., .3
    for _ in range(30):
        mid = (low+high)/2
        distances = []
        for i, (_,_,_,src) in enumerate(val):
            p = src.view(1,12) + mid*noise[i]
            p[:,6:] = torch.minimum(torch.maximum(p[:,6:],drive_lo),drive_hi)
            distances.append(pair_l2(p))
        if sum(distances)/len(distances) < target:
            low = mid
        else:
            high = mid
    sigma = (low+high)/2
    by_kind = {(kind,gen,h):[] for kind in ('pd','random') for gen in SOURCES for h in range(8)}
    metrics = {'pd':{},'random':{}}
    for gen, source in SOURCES.items():
        rows = torch.load(source,map_location='cpu')
        eval_rows = proposals(rows,test_names,model,finger_names,lower,upper)
        assert len(eval_rows) == 200 and {obj for obj,*_ in eval_rows} == test_names
        noise = torch.randn((200,8,12),generator=generator)
        for i,(obj,sid,pred,src) in enumerate(eval_rows):
            rand = src.reshape(1,12) + sigma*noise[i]
            rand[:,6:] = torch.minimum(torch.maximum(rand[:,6:],drive_lo),drive_hi)
            for kind, p in (('pd',pred),('random',rand)):
                metrics[kind].setdefault(gen,[]).append(pair_l2(p))
                for h in range(8):
                    by_kind[(kind,gen,h)].append({'object_name':obj,'source_grasp_index':sid,
                        'generator':'DRO' if gen=='dro' else 'GenDexGrasp','head_id':h,'params':p[h].clone()})
    for (kind,gen,h),seeds in by_kind.items():
        assert len(seeds) == 200
        torch.save({'info':{'model_sha256':digest,'kind':kind,'generator':gen,'head_id':h,'random_sigma':sigma if kind=='random' else None},
                    'seeds':seeds},OUT/f'{kind}_{gen}_head{h}.pt')
    summary = {'model_path':str(MODEL),'model_sha256':digest,'model_epoch':ckpt['epoch'],
        'validation_objects':sorted(val_names),'evaluation_objects':split['test'][:10],
        'validation_target_count':len(val),'learned_l2_val':target,'sigma_frozen_on_val':sigma,
        'seed':RNG_SEED,
        'evaluation_l2':{kind:{gen:sum(nums)/len(nums) for gen,nums in gens.items()} for kind,gens in metrics.items()}}
    (OUT/'seed_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
