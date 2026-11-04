"""Generate UTF-8 byte output with one prefill and carried minKron state."""
import argparse
import torch
from events import ByteBatch
from model import HATConfig,HATSHoE

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='results/language/checkpoint.pt')
    p.add_argument('--prompt',default='ROMEO:\n');p.add_argument('--count',type=int,default=240)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=123)
    args=p.parse_args();torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=HATSHoE(HATConfig(**saved['config'])).eval();model.load_state_dict(saved['model'])
    result,state=model.generate(ByteBatch.text(args.prompt),args.count,args.temperature,args.seed)
    print(bytes(result[0].tolist()).decode('utf-8',errors='replace'))
    print(f'\nCarried state: {state.storage_bytes()} bytes; no KV cache.')
