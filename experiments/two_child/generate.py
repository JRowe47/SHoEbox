import argparse
import torch
from model import HierarchicalSHoE,HierarchyConfig

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='results/full/checkpoint.pt')
p.add_argument('--prompt',default='ROMEO:\n');p.add_argument('--characters',type=int,default=240)
p.add_argument('--temperature',type=float,default=.8);args=p.parse_args()
torch.set_num_threads(2)
saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
model=HierarchicalSHoE(HierarchyConfig(**saved['config'])).eval();model.load_state_dict(saved['model'])
chars=saved['chars'];tokens=torch.tensor([[chars.index(c) for c in args.prompt]])
enabled=not saved['protocol']['args'].get('no_reinforcement',False)
result=model.generate(tokens,args.characters,args.temperature,reinforce=enabled)
print(''.join(chars[i] for i in result[0].tolist()))
