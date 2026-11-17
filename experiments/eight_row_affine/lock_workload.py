"""Identity-only exclusions and independent registered A11 byte-stream lock."""
import argparse,hashlib,importlib,json,random
from pathlib import Path
from experiments.selective_record_key_credit import tasks
from experiments.row_parallel_affine.lock_workload import CONDITIONS,reference_episode,canonical,sha

ROOT=Path(__file__).resolve().parents[2]
NAMES=('interference_retention','query_first_retention','chronological_retention','relative_chronological_retention','discriminative_keys','record_presence_cost','diverse_record_context','balanced_record_context')
def identities(vocab=3):
    for k0 in range(64):
        for k1 in range(k0+1,64):
            for v0 in range(vocab):
                for v1 in range(vocab):
                    if v0!=v1:yield canonical(['binding',[[k0,v0],[k1,v1]]])

def exclusion_manifest():
    """Read historical identity commitments; construct no audit World/episode."""
    modules=[importlib.import_module('experiments.'+name+'.tasks') for name in NAMES]
    prior=[importlib.import_module('experiments.'+name) for name in ('mad_next.tasks','factorized_readout.tasks','query_access.tasks','constituent_readout.tasks','constituent_readout.tasks_v3','joint_readout.tasks','joint_switch.tasks')]
    salts=[getattr(m,'SALT',b'EXP00-v2\0') for m in prior]
    def historical(key):
        return json.loads(key)[1][1][0]<48 and any(int.from_bytes(hashlib.sha256(s+key).digest()[:8],'big')%10>=8 for s in salts)
    reserved=set();by_stage=[];proof=[]
    for i,module in enumerate(modules,1):
        path=ROOT/('reports/exp0305/evidence/fit_plan.json' if i==1 else f'reports/exp0305a{i}/evidence/prefit.json')
        folds=json.loads(path.read_text())['folds'];stage=set()
        for split,n,vocab in ([('audit',256,2)] if i==1 else [('audit',256,3),('dev',128,3),('dev',128,2)]):
            values=[]
            for key in identities(vocab):
                b=int.from_bytes(hashlib.sha256(module.SALT+key).digest()[:8],'big')%10
                part='train' if b<6 else 'dev' if b<8 else 'audit'
                if part==split and key not in reserved and not historical(key):values.append(key)
            random.Random(20261122+i).shuffle(values);selected=values[:n];assert len(selected)==n
            label='binary_dev' if i>1 and vocab==2 else split
            assert sha(b'\n'.join(selected))==folds[label]['sha256'],(i,label)
            proof.append(dict(stage=i,split=label,count=n,sha256=folds[label]['sha256'],manifest=str(path.relative_to(ROOT)),manifest_sha256=sha(path.read_bytes())))
            stage.update(selected)
        reserved.update(stage);by_stage.append(frozenset(stage))
    # The original function executes unchanged with its identity suppliers bound
    # to the manifests just verified, so it cannot instantiate an audit World.
    for i,values in enumerate(by_stage,1):setattr(tasks,'reserved_a'+str(i),lambda values=values:values)
    tasks.excluded.cache_clear()
    for key in identities():assert tasks.excluded(key)==(historical(key) or key in reserved)
    return historical,reserved,proof

def main():
    p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    archived=json.loads((a.prior/'pre_fit_worlds.json').read_text())
    primary={canonical(x) for values in archived['worlds'].values() for x in values};binary={canonical(x) for x in archived['binary_dev']}
    assert len(primary)==896 and len(binary)==128
    old=json.loads((ROOT/'reports/exp0305a10/timing_worlds.lock.json').read_text());a10={canonical(x) for x in old['ordered_worlds']};assert len(a10)==16
    historical,reserved,proof=exclusion_manifest()
    previous=[key for key in identities() if historical(key) or key in reserved or key in primary or key in binary]
    assert sha(b'\n'.join(previous))==old['exclusion_sha256'] and len(previous)==old['excluded_groups']
    excluded=set(previous)|a10;eligible=[key for key in identities() if key not in excluded]
    independent=[canonical(['binding',[[x,v],[y,u]]]) for x in range(64) for y in range(x+1,64) for v in range(3) for u in range(3) if v!=u]
    independent=[key for key in independent if not tasks.excluded(key) and key not in primary|binary|a10]
    assert eligible==independent and len(eligible)>=16
    enumeration=sha(b'\n'.join(eligible));random.Random(20261203).shuffle(eligible)
    # Exactly one registered random shuffle. Independent check reproduces the
    # Fisher-Yates operations with the same RNG, rather than drawing a new sample.
    rng=random.Random(20261203)
    for j in range(len(independent)-1,0,-1):
        k=rng._randbelow(j+1);independent[j],independent[k]=independent[k],independent[j]
    selected=eligible[:16];assert eligible==independent
    data=[]
    for condition in CONDITIONS:
        for i,key in enumerate(selected):
            w=tasks.World('binding',tuple(map(tuple,json.loads(key)[1])))
            for variant in range(4):
                actual=tasks.make(w,condition,variant,0,(0,0),i%2).ep.data
                expected=reference_episode(key,condition,variant,i%2);assert actual==expected
                data.append(actual)
    raw=b''.join(data);(a.out/'timing.bin').write_bytes(raw)
    result=dict(passed=True,eligible_worlds=len(eligible),enumeration_sha256=enumeration,
        ordered_worlds=[json.loads(x) for x in selected],ordered_group_sha256=[sha(x) for x in selected],
        ordered_sample_sha256=sha(b'\n'.join(selected)),excluded_groups=len(excluded),
        exclusion_sha256=sha(b'\n'.join(key for key in identities() if key in excluded)),
        a9_primary_worlds=896,f1_binary_worlds=128,a10_timing_worlds=16,identity_manifest_proofs=proof,
        stream_sha256=sha(raw),stream_length=len(raw),padding_bytes=raw.count(9),nuisance_bytes=sum(b>=160 for b in raw),
        episodes=384,conditions=CONDITIONS,offset=0,spacers=[0,0],updated='world_index % 2',rng=20261203,
        prior_world_file_sha256=sha((a.prior/'pre_fit_worlds.json').read_bytes()),
        research_fitting_bytes=0,model_calls=0,audit_episodes=0,excluded_world_objects=0,accuracy_scored=False,opened_for_future_exclusion=True)
    (a.out/'lock.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('ordered_worlds','ordered_group_sha256','identity_manifest_proofs')},indent=2))
if __name__=='__main__':main()
