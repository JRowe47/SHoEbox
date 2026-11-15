import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from common import Model,runtime,build,install,atomic,file_hash,take_documents

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
work=a.base/'diagnostic_40m';build(work/'runtime')
manifest=json.loads((work/'data/manifest.json').read_text());all_docs=manifest['files']['train']['documents']
train=take_documents(all_docs,3_000_000);remaining=[d for d in all_docs if d['row_idx']>train[-1]['row_idx']]
screen=take_documents(remaining,524288);audit_candidates=[d for d in remaining if d['row_idx']>screen[-1]['row_idx']]
audit=take_documents(audit_candidates,1_000_000)
assert not {d['id'] for d in train}&{d['id'] for d in screen+audit}
assert not {d['id'] for d in screen}&{d['id'] for d in audit}
parent=a.base/'run_20m_40m/latest.shoe'
protocol=dict(parent_sha256=file_hash(parent),training_data_sha256=file_hash(work/'data/train.bin'),
 train_documents=train,fresh_screen_documents=screen,fresh_audit_documents=audit,
 development_documents=manifest['files']['development']['documents'],
 screen_targets=1_000_000,confirmation_targets=3_000_000,seed=7,chunk_targets=4096,
 selection='Inspect all nine one-factor/rate arms at 1M; continue baseline and at most two promising arms to 3M. Independent 1M-byte audit is scored only at confirmation.',
 rates='Preserve current values unless changed in the named arm. Freeze-bias retains existing bias values; no-aux disables contrastive, reconstruction, and SDR predictive losses.')
atomic(work/'protocol.json',protocol)
data=np.fromfile(work/'data/train.bin',np.uint8)[:8193]
with Model.load(parent) as original:
    original.reset_context(clear_memory=True);loss1=original.run(data[:-1],data[1:]);original.finish();digest1=hashlib.sha256(original.dumps()).hexdigest()
base,lib=install(work/'runtime')
with Model.load(parent) as diagnostic:
    diagnostic.reset_context(clear_memory=True);loss2=diagnostic.run(data[:-1],data[1:]);diagnostic.finish();digest2=hashlib.sha256(diagnostic.dumps()).hexdigest()
assert loss1==loss2 and digest1==digest2
atomic(work/'runtime_verification.json',dict(baseline_parity=True,targets=8192,nll=loss1,state_sha256=digest1,parent_sha256=protocol['parent_sha256']))
print(json.dumps(dict(ready=True,parity=True,training_targets=sum(d['targets'] for d in train),screen_targets=sum(d['targets'] for d in screen),audit_targets=sum(d['targets'] for d in audit))))
