"""Export the preselected seed-7 model with a persistent v10 policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shoe_multiscale import Model
from experiments.scale_100m.train import frozen_score,file_hash,atomic

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
p.add_argument('--audit-data',type=Path,required=True);p.add_argument('--audit-protocol',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--cpu',type=int,default=8);a=p.parse_args();os.sched_setaffinity(0,{a.cpu});a.out.mkdir(parents=True,exist_ok=True)
source=a.work/'D_warmup_only_s7/latest.shoe';expected=json.loads((a.work/'D_warmup_only_s7/probe.json').read_text())
reference=json.loads((a.work/'export_reference.json').read_text());meta=json.loads((a.data/'manifest.json').read_text())
path=a.out/'SHoEbox_5M_FineWeb_20M.shoe'
with Model.load(source) as m:
    before=m.dumps()[56:];m.set_balance_policy('warmup')
    assert m.dumps()[56:]==before
    assert m.stats()['checkpoint_version']==10 and m.stats()['balance_policy']=='warmup'
    assert m.routing_stats()['levels'][0]['events']==20_000_000
    m.save(path)
digest=file_hash(path)
with Model.load(path) as m:
    dev=np.fromfile(a.data/'development.bin',np.uint8)
    nll,_=frozen_score(m,dev,meta['files']['development']['documents'])
    audit=np.fromfile(a.audit_data,np.uint8);docs=json.loads(a.audit_protocol.read_text())['fresh_audit_documents']
    audit_loss,_=frozen_score(m,audit,docs)
    assert nll==expected['frozen']['full']['nll'] and audit_loss==expected['audit']['nll']
    stats=m.stats();x=np.fromfile(a.data/'train.bin',np.uint8)[:4098]
    loss=m.run(x[:-1],x[1:]);payload=hashlib.sha256(m.dumps()[56:]).hexdigest()
    assert loss==reference['continuation_nll'] and payload==reference['continuation_payload_sha256']
assert file_hash(path)==digest
result=dict(file=path.name,checkpoint_sha256=digest,source_sha256=file_hash(source),
    development_nll=nll,audit_nll=audit_loss,stats=stats,training_targets=20_000_000,seed=7,
    learned_payload_preserved=True,development_and_audit_exact=True,continuation_exact=True,
    continuation_targets=4097,continuation_nll=loss,continuation_payload_sha256=payload,
    diagnostic_globals_required=False)
atomic(a.work/'export_verification.json',result)
print(json.dumps(result,indent=2))
