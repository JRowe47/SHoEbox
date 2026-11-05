import json, random, sys, time
from pathlib import Path
import torch
from experiments.text_regions.composition_train import update, save_checkpoint, source_hashes
from experiments.text_regions.composition_data import PERMUTATIONS,pairs,batch_from_keys,training_batch
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.hat_minkron.train import optimizer_for
torch.set_num_threads(2);torch.manual_seed(20261105)
model=RegionLanguageModel(GraphConfig.tiny(readout_mode='concat',rounds=2,recurrent_feedback=True,evidence_gated_ports=True,feedback_gate_init=.1))
optimizer=optimizer_for(model,.003);rng=random.Random(20261105)
protocol=dict(lr=.003,warmup=20,rounds=2,feedback=True,source_hashes=source_hashes())
copies=[(a,b) for a,b in pairs('train') if all(PERMUTATIONS[b][PERMUTATIONS[a][q]]==q for q in range(4))]
history=[];counters={}
out=Path('artifacts/composition_copy_probe');out.mkdir(exist_ok=True)
for step in range(600):
    if step<100:
        keys=[]
        for _ in range(16):
            a,b=rng.choice(copies);keys.extend((a,b,q) for q in range(4))
        rng.shuffle(keys);batch=batch_from_keys(keys)
    else:batch=training_batch(rng,64,lookup_warmup=step<300)
    start=time.perf_counter();row=update(model,optimizer,batch,step,protocol);row['seconds']=time.perf_counter()-start;history.append(row)
    if (step+1)%25==0:print(json.dumps(row),flush=True)
    if (step+1)%50==0:
        save_checkpoint(out/'last.pt',model,optimizer,rng,step+1,protocol,0.,counters,history)
        (out/'history.json').write_text(json.dumps(history))
