"""Audit and summarize the completed compact memory-context acceptance run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_memory_selectivity import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/memory_context')
    p.add_argument('--results',type=Path,default=ROOT/'experiments/text_regions/results/memory_context')
    p.add_argument('--tests',type=Path,required=True)
    a=p.parse_args();r=a.root
    evaluations={name:json.loads((r/f'{name}_evaluation.json').read_text()) for name in ('parent','initial','final')}
    audit=json.loads((r/'source_audit.json').read_text());resume=json.loads((r/'resume_checks.json').read_text())
    digest=hashlib.sha256((r/'cohorts.json').read_bytes()).hexdigest()
    expected={'pairs_0','pairs_32','pairs_64','single_64','correction_64','language'}
    for name,evaluation in evaluations.items():
        if set(evaluation['measurements'])!=expected or evaluation['identity']['cohorts_sha256']!=digest:
            raise ValueError('Incomplete or mismatched evaluation.')
        actual=hashlib.sha256((r/('parent.pt' if name=='parent' else f"pilot/{'initial' if name=='initial' else 'last'}.pt")).read_bytes()).hexdigest()
        if evaluation['identity']['checkpoint_sha256']!=actual:raise ValueError('Checkpoint mismatch.')
    if audit['source_mismatches'] or not resume['exact_resume']:raise ValueError('Source/resume validation failed.')
    tests=a.tests.read_text()
    if 'Ran 94 tests' not in tests or not tests.rstrip().endswith('OK'):raise ValueError('Incomplete test suite.')
    history=[json.loads(line) for line in (r/'pilot/history.jsonl').read_text().splitlines()]
    if [x['step'] for x in history]!=list(range(1,audit['completed_steps']+1)):raise ValueError('Incomplete continuation.')
    scores={name:{key:{k:value[k] for k in ('both_correct','conflicting','accuracy','nll') if k in value}
                       for key,value in evaluation['measurements'].items()} for name,evaluation in evaluations.items()}
    result=dict(phase='memory_context_v1',status='implemented; compact acceptance complete; no model promotion',
        source_audit=audit,resume=resume,tests=94,metrics=scores,
        source_intervention=evaluations['final']['measurements']['pairs_0']['source_intervention'],
        memory_work=evaluations['final']['measurements']['pairs_64']['memory_work'],
        training_seconds=sum(x['seconds'] for x in history),peak_mib=max(x['peak_mib'] for x in history),
        cohort_sha256=digest)
    write_json(r/'summary.json',result)
    a.results.mkdir(parents=True,exist_ok=True)
    for name in ('parent_evaluation.json','initial_evaluation.json','final_evaluation.json','cohorts.json',
                 'source_audit.json','resume_checks.json','summary.json'):
        shutil.copy2(r/name,a.results/name)
    for name in ('history.jsonl','protocol.json'):shutil.copy2(r/'pilot'/name,a.results/name)
    shutil.copy2(a.tests,a.results/'tests.txt')
    print(json.dumps(dict(tests=94,steps=audit['completed_steps'],metrics=scores)))


if __name__=='__main__':main()
