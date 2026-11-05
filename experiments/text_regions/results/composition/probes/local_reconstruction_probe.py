import json, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from experiments.text_regions.composition_train import source_hashes
from experiments.text_regions.composition import encode_sources, answer_query, evaluate
from experiments.text_regions.composition_data import PERMUTATIONS,pairs,batch_from_keys,training_batch
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.hat_minkron.train import optimizer_for
torch.set_num_threads(2);torch.manual_seed(20261105)
model=RegionLanguageModel(GraphConfig.tiny(readout_mode='concat',rounds=2,recurrent_feedback=True,evidence_gated_ports=True,feedback_gate_init=.1))
head=nn.Linear(32,16)
whole=nn.ModuleDict(dict(graph=model,reconstruction=head))
optimizer=optimizer_for(whole,.003);rng=random.Random(20261105)
copies=[(a,b) for a,b in pairs('train') if all(PERMUTATIONS[b][PERMUTATIONS[a][q]]==q for q in range(4))]
history=[];out=Path('artifacts/composition_reconstruction_probe');out.mkdir(exist_ok=True)
for step in range(800):
    if step<100:
        keys=[]
        for _ in range(16):
            a,b=rng.choice(copies);keys.extend((a,b,q) for q in range(4))
        rng.shuffle(keys);batch=batch_from_keys(keys)
    else:batch=training_batch(rng,64,lookup_warmup=step<300)
    start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:group['lr']=.003*min(1.,(step+1)/20)
    prefix=encode_sources(model,batch);query=answer_query(model,batch,prefix['state'],2)
    ce=F.cross_entropy(query['logits'][:,-1],batch.answers)
    sources=torch.cat([r['state'].ports.latest[n].payload for r in (prefix,query) for n in ('B','C')])
    targets=torch.cat([batch.first,batch.second,batch.first,batch.second])-65
    reconstructed=head(sources).reshape(-1,4)
    local=F.cross_entropy(reconstructed,targets.flatten())
    loss=ce+.5*local+.005*(prefix['aux']+query['aux'])+.0005*(prefix['hat_aux']+query['hat_aux'])
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(whole.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    row=dict(step=step+1,nats=float(ce.detach()),local_nats=float(local.detach()),local_accuracy=float(reconstructed.argmax(-1).eq(targets.flatten()).float().mean()),accuracy=float(query['logits'][:,-1].argmax(-1).eq(batch.answers).float().mean()),gradient=float(grad),seconds=time.perf_counter()-start)
    history.append(row)
    if (step+1)%25==0:print(json.dumps(row),flush=True)
    if (step+1)%50==0:
        temp=out/'last.tmp';torch.save(dict(config=model.cfg.to_dict(),weights=model.state_dict(),source_head=head.state_dict(),optimizer=optimizer.state_dict(),step=step+1,history=history,torch_rng=torch.get_rng_state(),rng=rng.getstate(),source_hashes=source_hashes()),temp);temp.replace(out/'last.pt')
        (out/'history.json').write_text(json.dumps(history))
    if (step+1)%200==0:
        val=evaluate(model,count=128,rounds=(1,2,4))
        print(json.dumps({'step':step+1,'validation':{k:v['accuracy'] for k,v in val.items()}}),flush=True)
