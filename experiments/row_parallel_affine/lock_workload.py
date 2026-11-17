"""Independent fresh timing allocation and byte-for-byte fixture construction."""
import argparse,hashlib,importlib,json,random
from pathlib import Path
from experiments.selective_record_key_credit import tasks

CONDITIONS=('pad0','repeat16','diverse16','compete16','overwrite16','diverse256')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,separators=(',',':')).encode()

def reference_episode(key,condition,variant,updated):
    # Independent literal byte construction; never calls encode/make/interpreter.
    records=json.loads(key)[1];state=dict(records);order=[0,1] if variant<2 else [1,0]
    queries=[k for k,v in records][::(-1 if variant%2 else 1)]
    base=bytearray([1,16])
    for i in order:
        k,v=records[i];base.extend([2,96+k,3,32+v,4])
    # Original canonical query-tail padding is determined BEFORE inserting filler.
    tail=bytearray();length=len(base)
    for k in queries:
        tail.extend([5,96+k,3]);length+=3
        pad=(-length-1)%64;tail.extend([9]*pad);length+=pad
        tail.extend([6,32+state[k],4]);length+=3
    n=int(''.join(c for c in condition if c.isdigit()));kind=condition.rstrip('0123456789')
    rng=random.Random(int.from_bytes(hashlib.sha256(key+b'noise').digest()[:8],'big'))
    if kind=='pad':noise=bytes([9])*n
    elif kind=='repeat':noise=bytes([200])*n
    elif kind=='diverse':noise=bytes(rng.randrange(160,256) for _ in range(n))
    elif kind=='compete':
        keys=rng.sample([k for k in range(48) if k not in state],2)
        noise=bytes([2,96+keys[0],3,32,4,2,96+keys[1],3,33,4]+[9]*6)
    elif kind=='overwrite':
        k,_=records[updated];state[k]=3-sum(v for _,v in records)
        noise=bytes([2,96+k,3,32+state[k],4]+[9]*11)
        # Replace the two supplied input payloads exactly as the frozen fixture.
        which=0
        for i in range(len(tail)-1):
            if tail[i]==6:tail[i+1]=32+state[queries[which]];which+=1
    return bytes(base)+noise+bytes(tail)

def main():
    p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    archived=json.loads((a.prior/'pre_fit_worlds.json').read_text())
    primary={canonical(x) for values in archived['worlds'].values() for x in values}
    binary={canonical(x) for x in archived['binary_dev']};assert len(primary)==896 and len(binary)==128
    modules=('interference_retention','query_first_retention','chronological_retention','relative_chronological_retention','discriminative_keys','record_presence_cost','diverse_record_context','balanced_record_context')
    reserved=set()
    for i,name in enumerate(modules,1):
        m=importlib.import_module('experiments.'+name+'.tasks')
        for split,n,v in ([('audit',256,2)] if i==1 else [('audit',256,3),('dev',128,3),('dev',128,2)]):
            reserved.update(w.group_key for w in (m.worlds(split,n) if i==1 else m.worlds(split,n,vocab=v)))
    folds=[importlib.import_module('experiments.'+name).fold for name in ('mad_next.tasks','factorized_readout.tasks','query_access.tasks','constituent_readout.tasks','constituent_readout.tasks_v3','joint_readout.tasks','joint_switch.tasks')]
    eligible=[];independent=[];excluded_keys=[]
    for k0 in range(64):
        for k1 in range(k0+1,64):
            for v0 in range(3):
                for v1 in range(3):
                    if v0==v1:continue
                    key=canonical(['binding',[[k0,v0],[k1,v1]]])
                    old=key in reserved or (k1<48 and any(f(key)=='test' for f in folds))
                    assert old==tasks.excluded(key)
                    if old or key in primary or key in binary:excluded_keys.append(key)
                    else:independent.append(key)
                    if not tasks.excluded(key) and key not in primary and key not in binary:eligible.append(key)
    assert eligible==independent and len(eligible)>=16
    enumeration=sha(b'\n'.join(eligible));random.Random(20261202).shuffle(eligible)
    random.Random(20261202).shuffle(independent);selected=eligible[:16];assert selected==independent[:16]
    data=[];reference=[]
    for cond in CONDITIONS:
        for i,key in enumerate(selected):
            w=tasks.World('binding',tuple(map(tuple,json.loads(key)[1])))
            for variant in range(4):
                data.append(tasks.make(w,cond,variant,0,(0,0),i%2).ep.data)
                reference.append(reference_episode(key,cond,variant,i%2))
                assert data[-1]==reference[-1],(cond,i,variant)
    data=b''.join(data);assert data==b''.join(reference)
    (a.out/'timing.bin').write_bytes(data)
    result=dict(passed=True,eligible_worlds=len(eligible),enumeration_sha256=enumeration,
        ordered_worlds=[json.loads(x) for x in selected],ordered_group_sha256=[sha(x) for x in selected],
        ordered_sample_sha256=sha(b'\n'.join(selected)),excluded_groups=len(excluded_keys),
        exclusion_sha256=sha(b'\n'.join(excluded_keys)),a9_primary_worlds=896,f1_binary_worlds=128,
        stream_sha256=sha(data),stream_length=len(data),padding_bytes=data.count(9),
        nuisance_bytes=sum(x>=160 for x in data),episodes=384,conditions=CONDITIONS,
        offset=0,spacers=[0,0],updated='world_index % 2',rng=20261202,
        prior_world_file_sha256=sha((a.prior/'pre_fit_worlds.json').read_bytes()),
        research_fitting_bytes=0,model_calls=0,audit_episodes=0,accuracy_scored=False,
        opened_for_future_exclusion=True)
    (a.out/'lock.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('ordered_worlds','ordered_group_sha256')},indent=2))
if __name__=='__main__':main()
