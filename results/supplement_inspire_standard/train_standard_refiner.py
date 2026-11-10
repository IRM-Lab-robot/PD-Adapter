#!/usr/bin/env python3
"""Object-disjoint Inspire teacher assembly and fresh K=8 training under six-force battery."""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/user/yangbo_program/new')
OUT = ROOT / 'results/supplement_inspire_standard'
DRO = Path('/home/user/yangbo_program/DRO-Grasp')
ORIGINAL = ROOT / 'archive_old_unused/src_old_versions/train_pi_phi_v5_multihead.py'


def load_training_module():
    spec = importlib.util.spec_from_file_location('standard_v5_refiner', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assemble():
    split = json.loads((ROOT / 'eval/inspire_train_test_split.json').read_text())
    train30 = {r['object_name'] for r in torch.load(ROOT / 'eval/original_train30_zero_shot_b20.pt', map_location='cpu')}
    val10 = set(split['train']) - train30
    test = set(split['test'])
    assert len(train30) == 30 and len(val10) == 10 and not (train30 & val10 or train30 & test or val10 & test)
    sets = {'train': [], 'val': []}
    counts = Counter()
    for gen in ('dro', 'gendex'):
        for obj in split['train']:
            p = OUT / 'teacher' / gen / (obj.replace('+', '_') + '.pt')
            if not p.exists():
                raise FileNotFoundError(p)
            artifact = torch.load(p, map_location='cpu')
            assert artifact['objects'] == [obj] and artifact['total_source_grasps'] == 20
            args = artifact['args']
            assert args['selection_battery'] == 'standard_six_force05_steps6'
            assert args['population'] == 64 and args['generations'] == 4
            row = artifact['rows'][0]
            q_seed = torch.as_tensor(row['q_seed'], dtype=torch.float32)
            assert q_seed.shape == (20, 18)
            key = 'train' if obj in train30 else 'val'
            dataset, name = obj.split('+', 1)
            points = torch.load(DRO / 'data/PointCloud/object' / dataset / f'{name}.pt', map_location='cpu')[:512, :3].float()
            for cand in row['passing_candidates']:
                assert cand.get('cpu_success') is True and cand.get('label_source') == 'cpu_final_trusted'
                sid = int(cand['source_grasp_index'])
                assert 0 <= sid < 20
                params = torch.as_tensor(cand['params'], dtype=torch.float32).view(12)
                q = torch.as_tensor(cand['q'], dtype=torch.float32).view(18)
                sets[key].append({'object_name': obj, 'generator': 'DRO' if gen == 'dro' else 'GenDexGrasp',
                    'split': key, 'source_grasp_index': sid, 'object_points': points,
                    'w_star': q_seed[sid, :6], 'q_star': q_seed[sid, 6:],
                    'w_elite': params[:6], 'u_elite': params[6:], 'q_elite': q[6:],
                    'score': float(cand.get('score', 0)), 'source_file': str(p),
                    'elite_rank': -1})
                counts[(key, gen)] += 1
    summary = {'train_objects': sorted(train30), 'val_objects': sorted(val10), 'test_objects': split['test'],
               'positive_candidates': {f'{k}/{g}': v for (k, g), v in sorted(counts.items())},
               'source_zero_positive': {k: sorted((train30 if k == 'train' else val10) - {r['object_name'] for r in v}) for k, v in sets.items()}}
    assert sets['train'] and sets['val']
    torch.save({'train': sets['train'], 'val': sets['val'], 'summary': summary}, OUT / 'teacher_object_split.pt')
    (OUT / 'teacher_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ('positive_candidates','source_zero_positive')}, indent=2), flush=True)


def train():
    raw = torch.load(OUT / 'teacher_object_split.pt', map_location='cpu')
    assert set(raw['summary']['train_objects']).isdisjoint(raw['summary']['val_objects'])
    assert set(raw['summary']['test_objects']).isdisjoint(raw['summary']['train_objects'] + raw['summary']['val_objects'])
    base = load_training_module()
    base.set_seed(20262040)
    q_lower, q_upper, u_lower, u_upper = base.v4_mod.load_joint_limits()
    cpu_coupling = base.v4_mod.InspireCoupling(q_lower, q_upper).cpu().eval()
    train_groups, train_stats = base.build_groups(raw['train'], cpu_coupling, u_lower, u_upper)
    val_groups, val_stats = base.build_groups(raw['val'], cpu_coupling, u_lower, u_upper)
    assert set(g['object_name'] for g in train_groups).isdisjoint(g['object_name'] for g in val_groups)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base.PiPhiV5MultiHead(base.v4_mod.InspireCoupling(q_lower, q_upper).to(device),
                                  u_lower, u_upper, base.DELTA_SCALE, k=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    train_loader = DataLoader(base.MultiEliteGroupDataset(train_groups), batch_size=32, shuffle=True,
                              collate_fn=base.collate_batch, num_workers=0)
    val_loader = DataLoader(base.MultiEliteGroupDataset(val_groups), batch_size=32, shuffle=False,
                            collate_fn=base.collate_batch, num_workers=0)
    train_dir = OUT / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    best, log = float('inf'), []
    for epoch in range(1, 201):
        tr = base.run_epoch(model, train_loader, device, opt)
        va = base.run_epoch(model, val_loader, device)
        record = {'epoch':epoch, 'train':tr, 'val':va}
        log.append(record)
        if va['loss'] < best:
            best = va['loss']
            base.save_checkpoint(train_dir / 'best_pi_phi_v5_multihead.pt', model, epoch, record)
        if epoch % 10 == 0:
            print(f'epoch={epoch} train={tr["loss"]:.6f} val={va["loss"]:.6f} best={best:.6f}', flush=True)
    (train_dir / 'train_log.json').write_text(json.dumps(log, indent=2))
    best_record = min(log, key=lambda r:r['val']['loss'])
    meta = {'train_group_count':len(train_groups),'val_group_count':len(val_groups),
            'train_positive_count':len(raw['train']),'val_positive_count':len(raw['val']),
            'train_objects':raw['summary']['train_objects'],'val_objects':raw['summary']['val_objects'],
            'test_objects_excluded':raw['summary']['test_objects'],
            'train_stats':train_stats,'val_stats':val_stats,
            'best_epoch':best_record['epoch'],'best_val_loss':best,
            'checkpoint':str(train_dir / 'best_pi_phi_v5_multihead.pt'),
            'battery':'standard_six_force05_steps6','seed':20262040,'K':8}
    (train_dir / 'metrics.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=('assemble','train'), required=True)
    args = ap.parse_args()
    (assemble if args.step == 'assemble' else train)()
