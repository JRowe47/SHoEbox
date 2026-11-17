"""Independent data-only A9-F1 proof, preserving the original blocked evidence."""
import argparse, hashlib, importlib, json, random
from pathlib import Path
from . import tasks
from .run import ROOT, sha, write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True)
    a=p.parse_args();w=a.workspace;prior=w/'prior_a9'
    assert not (w/'f1_allocation.json').exists()
    lock=json.loads((ROOT/'reports/exp0305a9/evidence/blocked_source.lock.json').read_text())
    for name,digest in lock['locked_external_files'].items():
        assert sha(prior/name)==digest,name
    old=json.loads((prior/'world_feasibility_data.json').read_text())
    modules=('interference_retention','query_first_retention','chronological_retention',
             'relative_chronological_retention','discriminative_keys','record_presence_cost',
             'diverse_record_context','balanced_record_context')
    reserved=set()
    for i,name in enumerate(modules,1):
        m=importlib.import_module('experiments.'+name+'.tasks')
        for part,n,v in ([('audit',256,2)] if i==1 else [('audit',256,3),('dev',128,3),('dev',128,2)]):
            ws=m.worlds(part,n) if i==1 else m.worlds(part,n,vocab=v)
            reserved.update(x.group_key for x in ws)
    folds=[importlib.import_module('experiments.'+name).fold for name in (
        'mad_next.tasks','factorized_readout.tasks','query_access.tasks',
        'constituent_readout.tasks','constituent_readout.tasks_v3',
        'joint_readout.tasks','joint_switch.tasks')]
    def canonical(k0,k1,v0,v1):
        return json.dumps(['binding',[[k0,v0],[k1,v1]]],separators=(',',':')).encode()
    def excluded(key):
        small=all(k<48 for k,v in json.loads(key)[1])
        return key in reserved or (small and any(f(key)=='test' for f in folds))
    primary={part:[] for part in ('train','dev','audit')}
    for k0 in range(64):
        for k1 in range(k0+1,64):
            for v0 in range(3):
                for v1 in range(3):
                    if v0==v1:continue
                    key=canonical(k0,k1,v0,v1)
                    assert excluded(key)==tasks.excluded(key)
                    n=int.from_bytes(hashlib.sha256(b'SHoEbox-EXP0305-selective-record-key-credit-v9\0'+key).digest()[:8],'big')%10
                    part='train' if n<6 else 'dev' if n<8 else 'audit'
                    assert tasks.fold(key)==part
                    if not excluded(key):primary[part].append(key)
    for part,n in [('train',512),('dev',128),('audit',256)]:
        random.Random(20261201).shuffle(primary[part]);primary[part]=primary[part][:n]
        assert primary[part]==[x.group_key for x in tasks.worlds(part,n)]
        assert [json.loads(x) for x in primary[part]]==old['primary'][part]
    selected_primary=set().union(*map(set,primary.values()))
    eligible=[]
    for k0 in range(64):
        for k1 in range(k0+1,64):
            for v0 in range(2):
                for v1 in range(2):
                    if v0==v1:continue
                    key=canonical(k0,k1,v0,v1)
                    if excluded(key) or key in selected_primary:continue
                    eligible.append(key)
    assert len(eligible)==336 and len(selected_primary)==896
    enumeration_hash=hashlib.sha256(b'\n'.join(eligible)).hexdigest()
    random.Random(20261201).shuffle(eligible);binary=eligible[:128]
    assert binary==[x.group_key for x in tasks.worlds('dev',128,vocab=2)]
    assert len(set(binary))==128 and not set(binary)&selected_primary
    assert not any(excluded(key) for key in binary)
    # Subsets consumed by offset/order/delay checks are exact prefixes.
    for n in (1,16,20,64,128):
        assert [x.group_key for x in tasks.worlds('dev',n,vocab=2)]==binary[:n]
    assert (w/'streams.json').read_bytes()==(prior/'streams.json').read_bytes()
    for args in [('train',128,64,2),('audit',128,64,2),('dev',129,64,2),('dev',128,65,2)]:
        try:tasks.worlds(*args)
        except ValueError:pass
        else:raise AssertionError(('binary allocation escaped development-only scope',args))
    try:tasks.salted_worlds('dev',128,vocab=2)
    except ValueError as e:assert str(e)=='only 96 dev worlds at pool 64'
    else:raise AssertionError('original allocation failure was not retained')
    explicit=dict(worlds={k:[json.loads(x) for x in v] for k,v in primary.items()},
                  binary_dev=[json.loads(x) for x in binary],manifest=tasks.manifest(),
                  binary_assignment='development-only A9-F1',audit_answers_evaluated=False)
    write(w/'pre_fit_worlds.json',explicit)
    result=dict(passed=True,amendment='A9-F1',eligible_binary_worlds=336,
        selected_binary_worlds=128,primary_worlds_preserved=896,
        enumeration_sha256=enumeration_hash,
        shuffled_eligible_sha256=hashlib.sha256(b'\n'.join(eligible)).hexdigest(),
        binary_ordered_sha256=hashlib.sha256(b'\n'.join(binary)).hexdigest(),
        primary_ordered_sha256={k:hashlib.sha256(b'\n'.join(v)).hexdigest() for k,v in primary.items()},
        world_file_sha256=sha(w/'pre_fit_worlds.json'),original_streams_byte_identical=True,
        streams_sha256=sha(w/'streams.json'),original_blocked_lock_sha256=sha(ROOT/'reports/exp0305a9/evidence/blocked_source.lock.json'),
        prior_receipts_preserved=len(lock['locked_external_files']),
        registration_sha256=sha(ROOT/'reports/exp0305a9/A9_F1_REGISTRATION.md'),
        source_sha256=sha(Path(__file__)),primary_overlap=0,historical_overlap=0,
        research_fitting_bytes=0,inference_bytes=0,audit_answers_evaluated=False,
        reference='Raw canonical JSON, independent complete historical exclusion union, literal primary salt, original loop order, independent seeded shuffle; exact ordered byte equality.')
    write(w/'f1_allocation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
