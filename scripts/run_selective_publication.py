"""Recoverable three-seed producer comparison and independent evaluations."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/selective_publication'
SEEDS = (20261110, 20261111, 20261112)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parent = OUT/'parent.pt'
    if not parent.exists(): shutil.copyfile(ROOT/'artifacts/gradient_horizon/full/best.pt', parent)
    document = ROOT/'docs/SHoEbox_Selective_Publication_Protocol.md'
    lock = dict(protocol_sha256=hashlib.sha256(document.read_bytes()).hexdigest(),
                parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(), steps=240,
                seeds=SEEDS, test_episodes=512)
    lock_path = OUT/'protocol_lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != json.loads(json.dumps(lock)):
            raise ValueError('Protocol changed after lock.')
    else: lock_path.write_text(json.dumps(lock, indent=2)+'\n')
    status_path = OUT/'run_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'completed': [], 'state': 'training'}
    def save():
        temp = status_path.with_suffix('.tmp'); temp.write_text(json.dumps(status, indent=2)+'\n'); temp.replace(status_path)
    def train(seed, policy):
        name = f'{policy}_{seed}'
        path = OUT/name
        command = [sys.executable, '-m', 'experiments.text_regions.publication_train', '--parent', str(parent),
                   '--out', str(path), '--seed', str(seed), '--policy', policy, '--steps', '240',
                   '--validate-every', '60']
        if (path/'last.pt').exists():
            history = json.loads((path/'history.json').read_text())
            if history[-1]['step'] == 240: return name
            command.append('--resume')
        path.mkdir(exist_ok=True)
        with (path/'train.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return name
    try:
        status['state'] = 'training'; save()
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(train, seed, policy) for seed in SEEDS for policy in ('learned', 'always')
                    if f'{policy}_{seed}' not in status['completed']]
            for job in as_completed(jobs):
                status['completed'].append(job.result()); save()
        status['state'] = 'trained'; save()
        def evaluate(name):
            path = OUT/name
            with (path/'evaluation.log').open('a') as log:
                subprocess.run([sys.executable, 'scripts/evaluate_selective_publication.py', str(path)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            return 'evaluation/'+name
        status['state'] = 'evaluating'; save()
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(evaluate, f'{policy}_{seed}') for seed in SEEDS for policy in ('learned','always')
                    if 'evaluation/'+f'{policy}_{seed}' not in status['completed']]
            for job in as_completed(jobs): status['completed'].append(job.result()); save()
        for name, command in (
                ('training_audit', ['scripts/audit_selective_publication.py']),
                ('evaluation_audit', ['scripts/audit_selective_publication.py', '--evaluations']),
                ('regression', ['scripts/check_all.py'])):
            if name in status['completed']: continue
            logpath = OUT/(name+'.log')
            if name == 'regression' and logpath.exists() and 'All five experiment suites passed.' in logpath.read_text():
                status['completed'].append(name); save(); continue
            status['state'] = name; save()
            with logpath.open('w') as log:
                subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            status['completed'].append(name); save()
        status['state'] = 'complete'; save()
        print(json.dumps(status), flush=True)
    except BaseException as error:
        status['state'] = 'failed'; status['error'] = repr(error); save(); raise


if __name__ == '__main__': main()
