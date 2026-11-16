"""Learnability control on the new fold; not resource-matched native inference."""
import argparse
from pathlib import Path
import torch
from .tasks import dataset,balanced_offsets
from experiments.mad_next.tasks import Spec
from experiments.mad_next.controls import PositiveControl,evaluate
from experiments.mad_next.run import train_reference
from experiments.query_readout.run import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=dict(order_seed=601,reference_policy=dict(attention_lr=.05,batch_size=32))
    write(a.out/'control_plan.json',dict(seeds=[1,7,19],bytes_per_seed=250000,configuration=config,
        scope='record-encoded learned lookup; task-alphabet outputs, not matched native byte model'))
    spec=Spec('binding',vocab=2);tr=balanced_offsets(dataset(2048,'train'));dv=balanced_offsets(dataset(512,'dev'));rows=[]
    for seed in (1,7,19):
        checkpoint=a.out/f'control_{seed}.pt';row=train_reference(spec,seed,tr,dv,config,checkpoint,250000)
        rows.append(row)
    # Test only after all control fits have finished at the locked budget.
    te=balanced_offsets(dataset(1024,'test'))
    for row in rows:
        model=PositiveControl(spec,row['seed'],lr=.05);model.loads((a.out/f"control_{row['seed']}.pt").read_bytes())
        row['test']=evaluate(model,te);row['no_query']=evaluate(model,te,'no_query')
    write(a.out/'positive_control.json',dict(rows=rows))

if __name__=='__main__':main()
