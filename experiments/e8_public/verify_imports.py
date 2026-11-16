"""Import real E8 snapshots and prove exact public/research cursor trajectories."""
import argparse
import hashlib
import io
import json
import struct
from pathlib import Path
import time
import zipfile

import numpy as np
from shoe_multiscale import Model
from multiscale.e8.import_checkpoint import convert
from experiments.joint_cold_gate.adapter import Adapter
from experiments.joint_cold_gate.stream import advance
from experiments.joint_cold_gate.resume import language_advance,task_advance,load_task
from experiments.joint_readout.regressions import Corpus
from .bridge import PublicCursorModel,ledger


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);start=time.process_time();rows=[];cost=ledger();research_cost=ledger()
    paths=list((a.workspace/'exp01e8-pilot').glob('*.qr'))
    for name in ('formal','1m','5m'):
        paths+=list((a.workspace/f'exp01e8-{name}').glob('*/*.qr'))
    paths+=list((a.workspace/'exp01e8-regressions/local_gate').glob('*/*.checkpoint'))
    manifest=json.loads((a.workspace/'e8-checkpoint-manifest.json').read_text())
    for record in manifest['checkpoints']:
        assert hashlib.sha256((a.workspace/record['file']).read_bytes()).hexdigest()==record['sha256']
        assert hashlib.sha256((a.workspace/record['cursor']).read_bytes()).hexdigest()==record['cursor_sha256']
    for path in sorted(paths):
        if not path.with_suffix('.cursor.json').exists():continue
        raw=path.read_bytes()
        payload=raw
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:payload=z.read('adapter.qr')
        if len(payload)<52 or struct.unpack_from('=Q',payload,32)[0]!=0x38455041514853:
            continue  # An explicit non-E8 identity retains its historical loader.
        core_size=struct.unpack_from('=Q',payload,44)[0]
        kind=struct.unpack_from('=i',payload,52+core_size+7605*4)[0]
        if kind!=4:continue  # Registered generic-gate control, not public E8.
        m,receipt=Model.import_e8_research(payload)  # Invalid eligible imports fail.
        m.close()
        relative=path.relative_to(a.workspace);dest=a.out/'checkpoints'/relative.with_suffix('.shoe')
        receipt=convert(path,dest)
        cursor=json.loads(path.with_suffix('.cursor.json').read_text())
        seed=int(path.parent.name) if path.parent.name.isdigit() else 7
        old,decoder=load_task(path,Adapter,seed)
        # Decoder copies originate from the original saved arrays, not reseeding.
        other,decoder1=load_task(path,Adapter,seed);other.close()
        other,decoder2=load_task(path,Adapter,seed);other.close()
        with Model.import_e8_research(payload)[0] as original:
            # Test serialized *public* artifact as well as in-memory import.
            if zipfile.is_zipfile(dest):
                with zipfile.ZipFile(dest) as z:
                    saved=z.read('model.shoe')
                    with zipfile.ZipFile(path) as oz:assert z.read('decoder.npz')==oz.read('decoder.npz')
            else:saved=dest.read_bytes()
            assert saved==original.dumps()
        whole=PublicCursorModel(Model.loads(saved),cost);split=PublicCursorModel(Model.loads(saved),cost)
        if path.suffix=='.checkpoint':kind=path.stem.rsplit('-',1)[0]
        elif 'retained' in path.name:kind='retention'
        else:kind='binding'
        corpus=Corpus(a.workspace/('exp01e7-data' if kind=='language' else 'exp01e-data')) if kind in ('language','retention') else None
        def step(model,state,count,head):
            if kind=='binding':return advance(model,state,count)[0]
            if corpus:return language_advance(model,state,count,corpus)
            return task_advance(model,head,state,count,kind)
        before=old.inventory()
        with old,whole,split:
            expected=step(old,cursor,1000,decoder)
            uninterrupted=step(whole,cursor,1000,decoder1)
            partial=step(split,cursor,371,decoder2)
            with PublicCursorModel(Model.loads(split.dumps()),cost) as resumed:
                continued=step(resumed,partial,629,decoder2)
                assert expected==uninterrupted==continued,relative
                assert whole.dumps()==resumed.dumps(),relative
                assert whole.dumps()[48:]==old.dumps()[40:],relative
                if decoder:
                    for key in decoder.arrays:
                        np.testing.assert_array_equal(decoder.arrays[key],decoder1.arrays[key])
                        np.testing.assert_array_equal(decoder.arrays[key],decoder2.arrays[key])
                with whole.model.inference_copy(cold=False) as public,old.clone() as research:
                    for byte in bytes(range(256)):
                        research.observe(byte)
                        np.testing.assert_array_equal(public.observe(byte),research.log_probs())
                after=old.inventory()
                research_cost['input_bytes']+=1000
                for k in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):
                    research_cost[k]+=after[k]-before[k]
        rows.append(dict(checkpoint=str(relative),kind=kind,receipt=receipt,
                         exact_payload=True,exact_cursor=True,full_256_byte_predictions=True,
                         decoder_exact=decoder is not None))
        print('verified',relative,flush=True)
    assert len(rows)==29, "qualified E8 import inventory changed"
    for k in ('input_bytes','byte_9_inputs','objective_targets'):
        assert cost[k]%2==0
        research_cost[k]=cost[k]//2
    write(a.out/'verification.json',dict(rows=rows,public_correctness=cost,research_correctness=research_cost,
          correctness_only_discarded_input_bytes=3000*len(rows),
          cpu_seconds=time.process_time()-start,new_research_fit_bytes=0,
          scope='1000 research versus 1000 public versus 371 + save/load + 629 public. Pinned train only; no audit access. byte_9_inputs includes literal 9 in generic data; binding padding reported separately by exact cursors.'))
    print('verified',len(rows),'public import/cursor pairs',flush=True)
if __name__=='__main__':main()
