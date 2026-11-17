"""Data-only independent proof of the unamended A9 binary-fold shortage."""
import hashlib,importlib,json,random,sys
from collections import Counter
from pathlib import Path
from . import tasks
from .run import sha,write,ROOT

def main():
    w=Path(sys.argv[1]);assert not (w/'fold_feasibility.json').exists()
    modules=['interference_retention','query_first_retention','chronological_retention','relative_chronological_retention','discriminative_keys','record_presence_cost','diverse_record_context','balanced_record_context']
    prior={}
    for i,name in enumerate(modules,1):
        m=importlib.import_module('experiments.'+name+'.tasks')
        specs=[('audit',256,2)] if i==1 else [('audit',256,3),('dev',128,3),('dev',128,2)]
        prior['a'+str(i)]=set()
        for part,n,vocab in specs:
            worlds=m.worlds(part,n) if i==1 else m.worlds(part,n,vocab=vocab)
            prior['a'+str(i)].update(x.group_key for x in worlds)
    old_folds=[importlib.import_module('experiments.'+name).fold for name in ('mad_next.tasks','factorized_readout.tasks','query_access.tasks','constituent_readout.tasks','constituent_readout.tasks_v3','joint_readout.tasks','joint_switch.tasks')]
    def historical(key):
        records=json.loads(key)[1]
        return all(k<48 for k,v in records) and any(f(key)=='test' for f in old_folds)
    remaining={};all_counts={};stage_counts={};full=[]
    for vocab in (2,3):
        available={'train':[],'dev':[],'audit':[]};before=Counter();stage=Counter()
        for ka in range(64):
            for kb in range(ka+1,64):
                for va in range(vocab):
                    for vb in range(vocab):
                        if va==vb:continue
                        key=json.dumps(['binding',[[ka,va],[kb,vb]]],separators=(',',':')).encode()
                        number=int.from_bytes(hashlib.sha256(b'SHoEbox-EXP0305-selective-record-key-credit-v9\0'+key).digest()[:8],'big')%10
                        part='train' if number<6 else 'dev' if number<8 else 'audit';before[part]+=1
                        reasons=[]
                        if historical(key):reasons.append('historical_opened_fold_union')
                        reasons.extend(name for name,keys in prior.items() if key in keys)
                        assert bool(reasons)==bool(tasks.excluded(key)) and part==tasks.fold(key)
                        if reasons:stage[reasons[0]]+=1
                        else:available[part].append(key)
                        if vocab==2:full.append(dict(group_key=json.loads(key),fold=part,exclusions=reasons))
        for part,keys in available.items():
            random.Random(20261201).shuffle(keys)
            got=[x.group_key for x in tasks.salted_worlds(part,len(keys),vocab=vocab)]
            assert got==keys
        remaining[str(vocab)]=available;all_counts[str(vocab)]=dict(before);stage_counts[str(vocab)]=dict(stage)
    primary={part:[x.group_key for x in tasks.worlds(part,n)] for part,n in [('train',512),('dev',128),('audit',256)]}
    excluded_primary=set(primary['train'])|set(primary['audit'])|set(primary['dev'])
    unused_binary=[k for keys in remaining['2'].values() for k in keys if k not in excluded_primary]
    counts={v:{part:len(keys) for part,keys in parts.items()} for v,parts in remaining.items()}
    assert counts['2']['dev']==96
    try:tasks.salted_worlds('dev',128,vocab=2)
    except ValueError as e:error=str(e)
    else:raise AssertionError('unregistered shortage bypass')
    raw=dict(primary={p:[json.loads(k) for k in keys] for p,keys in primary.items()},available_binary_dev=[json.loads(k) for k in remaining['2']['dev']],binary_universe=full,audit_answers_evaluated=False)
    write(w/'world_feasibility_data.json',raw)
    write(w/'fold_feasibility.json',dict(passed=False,reason='registered binary dev sample cannot be formed',required_binary_dev_worlds=128,available_binary_dev_worlds=96,shortfall=32,exception=error,
      salt_hex=tasks.SALT.hex(),world_rng=20261201,key_pool=64,counts_before_exclusions=all_counts,available_after_exclusions=counts,mutually_exclusive_first_exclusion_counts=stage_counts,
      prior_reserved_union_counts={k:len(v) for k,v in prior.items()},binary_eligible_outside_all_selected_primary_groups=len(unused_binary),
      primary_selected_counts={p:len(v) for p,v in primary.items()},available_binary_dev_sha256=hashlib.sha256(b'\n'.join(remaining['2']['dev'])).hexdigest(),
      data_sha256=sha(w/'world_feasibility_data.json'),source_sha256=sha(Path(__file__)),registration_sha256=sha(ROOT/'reports/exp0305a8/A9_REGISTRATION.md'),
      independent_check='Raw canonical JSON plus literal SHA256 split rule, independently reconstructed exclusions, exact seeded order equals production generator for every available world.',
      research_fitting_bytes=0,inference_bytes=0,audit_answers_evaluated=False))
    print(json.dumps(json.loads((w/'fold_feasibility.json').read_text()),indent=2))
if __name__=='__main__':main()
