#!/usr/bin/env python3
"""Offline, object-clustered comparison of the standard-battery Inspire experiment."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch

ROOT=Path('/home/user/yangbo_program/new')
OUT=ROOT/'results/supplement_inspire_standard'
OBJECTS=tuple(json.loads((ROOT/'eval/inspire_train_test_split.json').read_text())['test'][:10])
GENERATORS=('dro','gendex')
METHODS=('raw','naive_phi','g4','pop256','random9','pd9')
BASELINES=('g4','pop256','random9')


def arms(method):
    if method in ('g4','pop256'):
        return (method,)
    k='pd' if method=='pd9' else 'random'
    return ('default',*(f'{k}_head{i}' for i in range(8)))


def quantile(values,p):
    v=sorted(values)
    x=(len(v)-1)*p
    lo=int(x)
    return v[lo] + (v[min(lo+1,len(v)-1)]-v[lo])*(x-lo)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--no-padding',action='store_true')
    cli=ap.parse_args()
    label_dir='revalidation_unpadded' if cli.no_padding else 'revalidation'
    sources={'dro':ROOT/'eval/original_test10_zero_shot_b20.pt',
             'gendex':ROOT/'eval/gendex_inspire_mainset_grasps.pt'}
    seedinfo=json.loads((OUT/'seed_summary.json').read_text())
    assert seedinfo['model_sha256']==hashlib.sha256((OUT/'train/best_pi_phi_v5_multihead.pt').read_bytes()).hexdigest()
    teacher=json.loads((OUT/'teacher_summary.json').read_text())
    assert not (set(OBJECTS) & (set(teacher['train_objects'])|set(teacher['val_objects'])))
    result={'protocol':'standard_six_force05_steps6','tail_padded_to_batch_size':not cli.no_padding,'test_objects':OBJECTS,
            'checkpoint_sha256':seedinfo['model_sha256'],'sigma_validation_only':seedinfo['sigma_frozen_on_val'],
            'lines':{}}
    for gen in GENERATORS:
        rows=torch.load(sources[gen],map_location='cpu')
        assert [r['object_name'] for r in rows]==list(OBJECTS)
        assert all(torch.as_tensor(r['q_dro_order']).shape==(20,18) for r in rows)
        line={'methods':{},'per_object':{},'comparisons':{}}
        values={}
        for method in METHODS:
            values[method]={}
            counts=[0]*5
            cpu_search=0
            cpu_reval=0
            for obj in OBJECTS:
                if method not in ('raw','naive_phi'):
                    seed_q=None
                    for arm in arms(method):
                        source=OUT/'search'/gen/arm/f'{obj.replace("+","_")}.pt'
                        if not source.exists():
                            raise RuntimeError(f'Incomplete search {source}')
                        s=torch.load(source,map_location='cpu')
                        assert s['objects']==[obj] and s['total_source_grasps']==20
                        assert [r['grasp_index'] for r in s['rows'][0]['grasp_rows']]==list(range(20))
                        q=torch.as_tensor(s['rows'][0]['q_seed'])
                        assert q.shape==(20,18)
                        if seed_q is not None:
                            assert torch.equal(q,seed_q)
                        seed_q=q
                        args=s['args']
                        assert args['selection_battery']=='standard_six_force05_steps6'
                        assert args['max_grasps']==20 and args['zero_file']==str(sources[gen])
                        assert args['population']*args['generations']==(256 if method in ('g4','pop256') else 32)
                        assert args['restarts']==1 and args['keep_per_grasp']==3 and args['cpu_final_candidates']==240
                        if arm.startswith(('pd_head','random_head')):
                            expected=OUT/f'{"pd" if arm.startswith("pd_head") else "random"}_{gen}_head{arm.rsplit("head",1)[1]}.pt'
                            assert args['warm_start_file']==str(expected)
                        else:
                            assert args['warm_start_file'] is None
                        cpu_search+=s['rows'][0]['cpu_final_evaluated']
                five=[]
                for ci in range(5):
                    p=OUT/label_dir/gen/method/f'{obj.replace("+","_")}_cond{ci}.json'
                    if not p.exists():
                        raise RuntimeError(f'Incomplete revalidation {p}')
                    rec=json.loads(p.read_text())
                    assert rec['generator']==gen and rec['method']==method and rec['object_name']==obj and rec['cond']==ci
                    assert rec['battery']=='standard_six_force05_steps6' and rec['threshold']==.02
                    assert rec['tail_padded_to_batch_size']==(not cli.no_padding) and rec['batch_size']==(32,64,128,256,96)[ci]
                    assert rec['success_sources']==len(rec['success_source_indices'])
                    assert set(rec['success_source_indices'])<=set(range(20))
                    counts[ci]+=rec['success_sources']
                    cpu_reval+=rec.get('cpu_candidates',20)
                    five.append(set(rec['success_source_indices']))
                values[method][obj]=[sum(i in s for s in five)/5 for i in range(20)]
            line['methods'][method]={'success_counts_conditions':counts,
                'mean_pct':sum(counts)/5/2,'std_conditions_pp':statistics.stdev(counts)/2,
                'search_samples_per_source':0 if method in ('raw','naive_phi') else (256 if method in ('g4','pop256') else 288),
                'search_cpu_final_per_source':cpu_search/200,'revalidation_cpu_candidates_total':cpu_reval}
        for obj in OBJECTS:
            line['per_object'][obj]={m:sum(values[m][obj])/20*100 for m in METHODS}
        for base in BASELINES:
            object_deltas=[sum(values['pd9'][obj][i]-values[base][obj][i] for i in range(20))/20*100 for obj in OBJECTS]
            rng=random.Random(20261221)
            boots=[sum(object_deltas[rng.randrange(10)] for _ in range(10))/10 for _ in range(5000)]
            lo,hi=quantile(boots,.025),quantile(boots,.975)
            line['comparisons'][f'pd9_vs_{base}']={'delta_pp':sum(object_deltas)/10,
                'cluster_95ci_pp':[lo,hi],'ci_excludes_zero':lo>0 or hi<0,
                'per_object_delta_pp':dict(zip(OBJECTS,object_deltas))}
        result['lines'][gen]=line
    path=OUT/('standard_unpadded_summary.json' if cli.no_padding else 'standard_summary.json')
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({'output':str(path),'lines':result['lines']},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
