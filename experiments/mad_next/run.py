"""Plan, train, then audit EXP00. Test labels never drive calibration selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
import time

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '1'
import numpy as np
import torch

from experiments.mad.backend import BenchModel
from experiments.mad.run import evaluate as native_evaluate, train_fit
from experiments.mad.tasks import TASKS as LEGACY_TASKS, TaskSpec, dataset as legacy_dataset, fixed_table, solve as legacy_solve
from .calibration import decision
from .controls import PositiveControl, evaluate as reference_evaluate
from .measurements import native_categories
from .tasks import Spec, dataset, exhaustive_oracles, fingerprint, parse, reencode, shortcuts

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def provenance():
    paths = [*sorted(HERE.glob('*.py')), *sorted((ROOT/'experiments/mad').glob('*.py')),
             ROOT/'experiments/mad/native.cpp', ROOT/'multiscale/model.py',
             ROOT/'multiscale/native.cpp', *sorted((ROOT/'multiscale').glob('*.hpp'))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'dirty': bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
            'source_sha256': hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            'source_files': hashes, 'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'platform': platform.platform()}


def make_plan(args):
    cells = [{'id':f'binding_v{v}', 'spec':asdict(Spec('binding',vocab=v))} for v in (2,4,8)]
    cells += [{'id':t, 'spec':asdict(Spec(t,items=4,key_pool=4))} for t in ('initialize','set','copy','swap')]
    cells += [{'id':'one_hop', 'spec':asdict(Spec('one_hop',vocab=16,items=3,key_pool=16))}]
    if args.tasks:
        cells = [c for c in cells if c['id'] in args.tasks]
        if len(cells) != len(set(args.tasks)):
            raise ValueError('unknown/duplicate task setting')
    seeds = args.seeds or [1,7,19]
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate seeds')
    for cell in cells:
        task = cell['spec']['task']
        cell['train_examples'] = (2048 if task in ('binding','one_hop') else
                                  8192 if task == 'initialize' else 65536)
        cell['reference_budget'] = (args.attention_budget if task in ('binding','one_hop') else
                                    args.gru_budget or {'initialize':8000000,'set':32000000,
                                                        'copy':128000000,'swap':128000000}[task])
    plan = {'experiment':'EXP00','protocol_version':2,'cells':cells,'seeds':seeds,
            'parameters':500000,'native_budget':args.native_budget,
            'dev_examples':512,'test_examples':1024,
            'data_seed':20261115,'order_seed':1067,'workers':args.workers,
            'native_policy':{'lr_scale':1.,'auxiliary':.1,'steps':0},
            'reference_policy':{'attention_lr':.05,'gru_lr':.003,'batch_size':32},
            'selection':'development exact shortcut-normalized band [0.1,0.9]; positive control >=95% mean and >=90% every seed',
            'fallback':'first declared/easiest setting if no calibrated setting; diagnostic only',
            'split_scope':'held-out concrete initial states; all operations and explicit presentation/lineage variants grouped; not unseen program structures',
            'counterfactuals':['query order','within-family operation arguments','64 byte-boundary offsets'],
            'reference_comparison':'record-encoded learnability controls; different budgets and state; no architecture efficiency ranking'}
    plan['fits'] = len(cells)*len(seeds)*2
    plan['training_bytes'] = sum((plan['native_budget']+c['reference_budget'])*len(seeds) for c in cells)
    return plan


def native_metrics(row):
    return {'cases':row['cases'],'token_accuracy':row['task_accuracy'],
            'token_accuracy_mode':'teacher_forced',
            'exact_accuracy':row['exact_answer_accuracy'],
            'task_nll':row['task_nll'],'native_byte_accuracy':row['native_byte_accuracy'],
            'native_exact_accuracy':row['native_exact_accuracy'],
            'native_byte_nll':row['native_byte_nll'],
            'chance_normalized_token':row['chance_normalized_accuracy']}


def train_reference(spec, seed, train, dev, config, checkpoint, budget):
    lr = config['reference_policy']['attention_lr' if spec.task in ('binding','one_hop') else 'gru_lr']
    model = PositiveControl(spec,seed,lr=lr)
    row = {'seed':seed,'model':model.kind,'inventory':model.inventory(),'curve':[]}
    row['curve'].append({'bytes':0,'cpu_seconds':0.,'dev':reference_evaluate(model,dev)})
    rng, order = random.Random(config['order_seed']), []
    total = cpu = wall = labels = discarded = 0
    digest = hashlib.sha256()
    interval = max(1,budget//4)
    next_eval = interval
    while total < budget:
        batch = []
        while len(batch) < config['reference_policy']['batch_size'] and total < budget:
            if not order:
                order = list(range(len(train)))
                rng.shuffle(order)
            idx = order.pop()
            ep = train[idx]
            cost = len(ep.data)-1
            used = min(cost,budget-total)
            total += used
            digest.update(f'{idx}:{used};'.encode())
            if used == cost:
                batch.append(ep)
                labels += len(ep.answer)
            else:
                discarded += used
        if batch:
            c,w = time.process_time(),time.perf_counter()
            model.update(batch)
            cpu += time.process_time()-c
            wall += time.perf_counter()-w
        if total >= next_eval or total == budget:
            row['curve'].append({'bytes':total,'cpu_seconds':cpu,'dev':reference_evaluate(model,dev)})
            while next_eval <= total:
                next_eval += interval
    payload = model.dumps()
    Path(checkpoint).write_bytes(payload)
    restored = PositiveControl(spec,seed,lr=lr)
    restored.loads(payload)
    reload_metrics = reference_evaluate(restored,dev)
    if reload_metrics != row['curve'][-1]['dev']:
        raise AssertionError('reference checkpoint predictions changed')
    row.update({'bytes':total,'cpu_seconds':cpu,'wall_seconds':wall,'supervised_targets':labels,
                'partial_episode_bytes_without_update':discarded,'training_order_sha256':digest.hexdigest(),
                'checkpoint_reload_exact':True,'dev':row['curve'][-1]['dev']})
    return row


def init_worker(queue):
    cpu = queue.get()
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def job(cell, seed, config, temporary):
    spec = Spec(**cell['spec'])
    train = dataset(spec,cell['train_examples'],'train',config['data_seed'])
    dev = dataset(spec,config['dev_examples'],'dev',config['data_seed'])
    stem = Path(temporary)/f"{cell['id']}_{seed}"
    npath,rpath = str(stem)+'.mad',str(stem)+'.pt'
    cfg = {'budget':config['native_budget'],'eval_every':max(1,config['native_budget']//4)}
    native = train_fit({'spec':asdict(spec.native()),'size':config['parameters'],'steps':0},
                       cfg,seed,1.,.1,train,dev,Path(npath))
    # Legacy learning curves are teacher forced. Select settings using a fresh
    # autoregressive evaluation, so prior answer labels cannot inflate the gate.
    with BenchModel.loads(Path(npath).read_bytes()) as final:
        native['dev'] = native_metrics(native_evaluate(final,dev,rollout=True))
    for point in native['curve']:
        point['dev'] = native_metrics(point['dev'])
        point['evaluation_mode'] = 'teacher_forced'
    native['model'] = 'corrected_C'
    reference = train_reference(spec,seed,train,dev,config,rpath,cell['reference_budget'])
    return {'cell':cell['id'],'seed':seed,'native':native,'reference':reference,
            'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'datasets':{'train_cases':len(train),'dev_cases':len(dev),
                        'train_sha256':fingerprint(train),'dev_sha256':fingerprint(dev),
                        'train_worlds':len({e.split_key for e in train}),
                        'dev_worlds':len({e.split_key for e in dev})},
            'shortcuts_dev':shortcuts(dev,spec),'checkpoints':{'native':npath,'reference':rpath}}


def calibrate(result):
    rows = []
    for cell in result['plan']['cells']:
        jobs = [r for r in result['runs'] if r['cell'] == cell['id']]
        d = decision([j['native'] for j in jobs],[j['reference'] for j in jobs],jobs[0]['shortcuts_dev'])
        rows.append({'cell':cell['id'],'task':cell['spec']['task'],**d})
    selected = []
    for task in dict.fromkeys(c['spec']['task'] for c in result['plan']['cells']):
        candidates = [r for r in rows if r['task'] == task and r['eligible_for_architecture_comparison']]
        if candidates:
            pick = min(candidates,key=lambda r:abs(r['shortcut_normalized_exact']-.5))
        else:
            pick = next(r for r in rows if r['task'] == task)
        selected.append({'task':task,'cell':pick['cell'],'diagnostic_only':not pick['eligible_for_architecture_comparison']})
    return {'settings':rows,'selected':selected,'selected_before_test':True}


def train(config,out):
    if out.exists():
        raise ValueError('output exists; use a new name or --phase audit')
    old_checks = 0
    for task in LEGACY_TASKS:
        s = TaskSpec(task)
        for part in ('train','dev','test'):
            for ep in legacy_dataset(s,32,part):
                if legacy_solve(ep,s,fixed_table(s,47)) != ep.answer:
                    raise AssertionError('legacy task oracle failure')
                old_checks += 1
    temp = tempfile.mkdtemp(prefix='shoebox-exp00-')
    result = {'status':'running','plan':config,'provenance':provenance(),
              'fixture_checks':{'exhaustive_tiny_oracles':exhaustive_oracles(),'legacy_nine_task_oracles':old_checks},
              'temporary_checkpoints':temp,'runs':[]}
    write_json(out,result)
    context = mp.get_context('spawn')
    queue = context.Queue()
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [0]
    workers = min(config['workers'],len(cpus))
    for cpu in cpus[:workers]:
        queue.put(cpu)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=init_worker,initargs=(queue,)) as pool:
        pending = {pool.submit(job,cell,seed,config,temp):(cell['id'],seed)
                   for cell in config['cells'] for seed in config['seeds']}
        for future in as_completed(pending):
            row = future.result()
            result['runs'].append(row)
            write_json(out,result)
            print(json.dumps({'finished':len(result['runs']),'jobs':len(pending),'cell':row['cell'],'seed':row['seed'],
                              'C_dev_exact':row['native']['dev']['exact_accuracy'],
                              'reference_dev_exact':row['reference']['dev']['exact_accuracy']}),flush=True)
    result['training_elapsed_wall_seconds'] = time.perf_counter()-start
    result['calibration'] = calibrate(result)
    result['status'] = 'trained_audit_pending'
    write_json(out,result)


def counterfactuals(model, cases, spec, evaluator):
    # One representative per initial-state group, not 128 presentation siblings.
    sample = list({e.split_key:e for e in cases}.values())[:128]
    output = {'query_reversed':evaluator(model,[reencode(e,spec,reverse_queries=True) for e in sample])}
    if spec.task in ('set','copy','swap'):
        output['operation_changed'] = evaluator(model,[reencode(e,spec,change_operation=True) for e in sample])
    offsets = []
    for offset in range(64):
        points = [reencode(e,spec,offset=offset) for e in sample[:32]]
        score = evaluator(model,points)
        offsets.append({'offset':offset,'cases':len(points),'exact_accuracy':score['exact_accuracy'],
                        'token_accuracy':score['token_accuracy']})
    output['offsets'] = offsets
    return output


def audit(out):
    result = json.loads(out.read_text())
    if result['status'] != 'trained_audit_pending':
        raise ValueError('audit requires completed training and a locked selection')
    result['audit_provenance'] = provenance()
    torch.set_num_threads(1)
    audits = []
    for selected in result['calibration']['selected']:
        cell = next(c for c in result['plan']['cells'] if c['id'] == selected['cell'])
        spec = Spec(**cell['spec'])
        cases = dataset(spec,result['plan']['test_examples'],'test',result['plan']['data_seed'])
        baseline = shortcuts(cases,spec)
        for row in [r for r in result['runs'] if r['cell'] == selected['cell']]:
            reference = PositiveControl(spec,row['seed'])
            reference.loads(Path(row['checkpoints']['reference']).read_bytes())
            with BenchModel.loads(Path(row['checkpoints']['native']).read_bytes()) as native:
                eval_c = lambda m,c:native_metrics(native_evaluate(m,c))
                entry = {'cell':cell['id'],'task':spec.task,'seed':row['seed'],
                         'diagnostic_only':selected['diagnostic_only'],'test_cases':len(cases),
                         'test_sha256':fingerprint(cases),'shortcuts_test':baseline,
                         'native':{**eval_c(native,cases),**native_categories(native,cases,spec)},
                         'reference':reference_evaluate(reference,cases),
                         'reference_lesions':{k:reference_evaluate(reference,cases,k) for k in
                                               (('no_query','zero_state','ignore_operation') if spec.task in ('set','copy','swap') else ('no_query','zero_state'))},
                         'native_counterfactuals':counterfactuals(native,cases,spec,eval_c),
                         'reference_counterfactuals':counterfactuals(reference,cases,spec,reference_evaluate)}
            audits.append(entry)
            result['audits'] = audits
            write_json(out,result)
            print(json.dumps({'audited':len(audits),'task':spec.task,'seed':row['seed'],
                              'C_test_exact':entry['native']['exact_accuracy'],
                              'reference_test_exact':entry['reference']['exact_accuracy']}),flush=True)
    shutil.rmtree(result.pop('temporary_checkpoints'))
    for row in result['runs']:
        row.pop('checkpoints')
    result['status'] = 'completed'
    result['production_promotion'] = 'none; EXP00 calibrates fixtures and controls'
    write_json(out,result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--phase',choices=['train','audit'],default='train')
    p.add_argument('--out',type=Path,default=Path('reports/exp00/results.json'))
    p.add_argument('--tasks',nargs='+')
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--native-budget',type=int,default=250000)
    p.add_argument('--attention-budget',type=int,default=250000)
    p.add_argument('--gru-budget',type=int,default=None,
                   help='override task-specific GRU budgets (8M/32M/128M/128M)')
    args = p.parse_args()
    if min(args.native_budget,args.attention_budget,args.gru_budget or 1,args.workers) < 1:
        p.error('budgets and workers must be positive')
    config = make_plan(args)
    if not args.execute:
        print(json.dumps(config,indent=2))
    elif args.phase == 'train':
        train(config,args.out)
    else:
        audit(args.out)


if __name__ == '__main__':
    main()
