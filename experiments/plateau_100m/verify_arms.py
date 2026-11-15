import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from common import Model,install,score,atomic,file_hash
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
work=a.base/'diagnostic_40m';_,lib=install(work/'runtime_followup')
protocol=json.loads((work/'protocol.json').read_text());selection=json.loads((work/'selection.json').read_text())
dev=np.fromfile(a.base/'data/development.bin',np.uint8);train=np.fromfile(work/'data/train.bin',np.uint8);out={}
for arm in selection['confirmed_arms']:
    path=work/'results'/arm/'latest.shoe';r=json.loads((path.parent/'results.json').read_text());before=file_hash(path)
    lib.diag_policy(int(arm!='zero_bias'),.1);lib.diag_aux_levels(1,1,1,1)
    with Model.load(path) as model:
        assert model.routing_stats()['levels'][0]['events']==43_000_000
        evaluation=score(model,dev,protocol['development_documents']);assert evaluation['nll']==r['development']['nll']
        with Model.loads(model.dumps()) as replay:
            model.reset_context(clear_memory=True);replay.reset_context(clear_memory=True)
            first=model.run(train[:1024],train[1:1025]);second=replay.run(train[:1024],train[1:1025])
            model.finish();replay.finish();assert first==second
            assert hashlib.sha256(model.dumps()).digest()==hashlib.sha256(replay.dumps()).digest()
    assert file_hash(path)==before==r['checkpoint_sha256']
    out[arm]=dict(reloaded_development_nll=evaluation['nll'],reload_exact=True,continuation_targets=1024,continuation_loss_and_state_exact=True,checkpoint_sha256=before,diagnostic_policy_restored=True)
    print(arm,json.dumps(out[arm]),flush=True)
atomic(work/'checkpoint_verification.json',out)
